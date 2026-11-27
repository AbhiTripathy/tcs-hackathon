"""
RushIQ Memory Store
===================
Lightweight, file-backed memory and context management for the RushIQ forecasting agent.

Memory layers
-------------
short_term  – current session scratch (in-memory only, cleared on restart)
long_term   – persisted restaurant config, preferences, operational rules
episodic    – persisted log of past predictions and their actual outcomes
semantic    – derived from model.json; menu knowledge, prep times, thresholds

Context retrieval is task-scoped and size-bounded so the agent never injects
irrelevant or excessive history into a prompt.

Design constraints
------------------
- Zero new runtime dependencies (stdlib json + pathlib only).
- Never overwrites model.json, backtest.json, or any training artefact.
- Memory write failures never crash the calling agent.
- Stale episodic records (> MAX_EPISODIC_DAYS old) are pruned on load.
- Duplicate episodic entries (same date + service) are deduplicated on write.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
MEMORY_DIR = ROOT / "memory" / "store"
MEMORY_DIR.mkdir(parents=True, exist_ok=True)

LONG_TERM_FILE = MEMORY_DIR / "long_term.json"
EPISODIC_FILE  = MEMORY_DIR / "episodic.json"
MODEL_FILE     = ROOT / "web" / "data" / "model.json"

MAX_EPISODIC_DAYS = 90          # prune records older than this
MAX_CONTEXT_EPISODIC = 10       # max recent episodic records injected into context
MAX_CONTEXT_BYTES = 8_000       # soft ceiling for context payload (chars ≈ bytes)

_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Short-term memory  (in-process, session-scoped)
# ---------------------------------------------------------------------------
class ShortTermMemory:
    """Holds the active request, task state, and intermediate results for one session."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def update(self, mapping: dict[str, Any]) -> None:
        self._data.update(mapping)

    def clear(self) -> None:
        self._data.clear()

    def snapshot(self) -> dict[str, Any]:
        return dict(self._data)

    def __repr__(self) -> str:
        return f"ShortTermMemory({list(self._data.keys())})"


# ---------------------------------------------------------------------------
# Helpers for JSON persistence
# ---------------------------------------------------------------------------
def _safe_read(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("memory read failed %s: %s", path, exc)
    return default


def _safe_write(path: Path, data: Any) -> bool:
    try:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        tmp.replace(path)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("memory write failed %s: %s", path, exc)
        return False


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Long-term memory  (restaurant config, preferences, rules)
# ---------------------------------------------------------------------------
_DEFAULT_LONG_TERM: dict[str, Any] = {
    "schema_version": 1,
    "restaurant": {
        "name": "RushIQ",
        "operating_hours": {"Lunch": "11:00–15:00", "Dinner": "18:00–23:00"},
        "services": ["Lunch", "Dinner"],
        "order_channels": ["Dine-in", "Takeaway", "Delivery"],
    },
    "preferences": {
        "shortage_penalty": 2,
        "waste_penalty": 1,
        "default_forecast_horizon_days": 7,
    },
    "operational_rules": [
        "Forecast beyond the historical range assumes stable weekly demand patterns.",
        "Calendar flags (holiday, event, promotion) must be supplied explicitly.",
        "Forecasts are unrounded until the prep optimizer chooses integers.",
        "Data is synthetic; item units proxy demand, unsold units proxy waste.",
    ],
    "updated_at": None,
}


class LongTermMemory:
    """Persisted restaurant configuration, user preferences, and operational rules."""

    def __init__(self, path: Path = LONG_TERM_FILE) -> None:
        self._path = path
        with _lock:
            self._data: dict[str, Any] = _safe_read(path, None)
            if self._data is None:
                self._data = dict(_DEFAULT_LONG_TERM)
                _safe_write(path, self._data)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> bool:
        with _lock:
            self._data[key] = value
            self._data["updated_at"] = _now_iso()
            return _safe_write(self._path, self._data)

    def update_preference(self, key: str, value: Any) -> bool:
        with _lock:
            prefs: dict = self._data.setdefault("preferences", {})
            prefs[key] = value
            self._data["updated_at"] = _now_iso()
            return _safe_write(self._path, self._data)

    def add_rule(self, rule: str) -> bool:
        with _lock:
            rules: list = self._data.setdefault("operational_rules", [])
            if rule not in rules:
                rules.append(rule)
                self._data["updated_at"] = _now_iso()
                return _safe_write(self._path, self._data)
        return False

    def snapshot(self) -> dict[str, Any]:
        return dict(self._data)


# ---------------------------------------------------------------------------
# Episodic memory  (prediction log with actual outcomes)
# ---------------------------------------------------------------------------
class EpisodicMemory:
    """
    Persisted log of past demand predictions and their recorded outcomes.

    Each record:
        date, service, predictions {item: float}, actual_demand {item: int},
        total_predicted, total_actual, mae, bias, flags, recorded_at

    Predictions and outcomes are kept separate. Storing a prediction does
    NOT imply it is a verified outcome; `record_outcome` updates only after
    real data is available.
    """

    def __init__(self, path: Path = EPISODIC_FILE) -> None:
        self._path = path
        with _lock:
            raw = _safe_read(path, {"schema_version": 1, "records": []})
            self._records: list[dict[str, Any]] = raw.get("records", [])
            self._prune_stale()

    # -- write ---------------------------------------------------------------

    def record_prediction(
        self,
        date: str,
        service: str,
        predictions: dict[str, float],
        flags: dict[str, int],
    ) -> bool:
        """Log a new forecast. Does not overwrite an existing record for same date+service."""
        with _lock:
            key = (date, service)
            existing = next((r for r in self._records if (r["date"], r["service"]) == key), None)
            if existing:
                logger.debug("episodic: prediction for %s %s already logged", date, service)
                return False
            total = sum(predictions.values())
            record = {
                "date": date,
                "service": service,
                "predictions": predictions,
                "total_predicted": total,
                "actual_demand": None,     # filled by record_outcome
                "total_actual": None,
                "mae": None,
                "bias": None,
                "flags": flags,
                "predicted_at": _now_iso(),
                "outcome_recorded_at": None,
            }
            self._records.append(record)
            return _safe_write(self._path, {"schema_version": 1, "records": self._records})

    def record_outcome(
        self,
        date: str,
        service: str,
        actual_demand: dict[str, int],
    ) -> bool:
        """Update an existing episodic record with verified actual demand."""
        with _lock:
            record = next(
                (r for r in self._records if r["date"] == date and r["service"] == service),
                None,
            )
            if record is None:
                logger.warning("episodic: no prediction found for %s %s to attach outcome", date, service)
                return False
            preds = record.get("predictions", {})
            errors = [abs(preds.get(k, 0) - v) for k, v in actual_demand.items()]
            biases = [(preds.get(k, 0) - v) for k, v in actual_demand.items()]
            record["actual_demand"] = actual_demand
            record["total_actual"] = sum(actual_demand.values())
            record["mae"] = float(sum(errors) / len(errors)) if errors else None
            record["bias"] = float(sum(biases) / len(biases)) if biases else None
            record["outcome_recorded_at"] = _now_iso()
            return _safe_write(self._path, {"schema_version": 1, "records": self._records})

    # -- read ----------------------------------------------------------------

    def recent(self, n: int = MAX_CONTEXT_EPISODIC) -> list[dict[str, Any]]:
        """Return the n most recent records (newest first)."""
        return list(reversed(self._records[-n:]))

    def for_date_service(self, date: str, service: str) -> dict[str, Any] | None:
        return next(
            (r for r in reversed(self._records) if r["date"] == date and r["service"] == service),
            None,
        )

    def average_mae(self, service: str | None = None) -> float | None:
        candidates = [
            r["mae"] for r in self._records
            if r["mae"] is not None and (service is None or r["service"] == service)
        ]
        return float(sum(candidates) / len(candidates)) if candidates else None

    def _prune_stale(self) -> None:
        cutoff = time.time() - MAX_EPISODIC_DAYS * 86400
        before = len(self._records)
        self._records = [
            r for r in self._records
            if _parse_epoch(r.get("predicted_at", "")) >= cutoff
        ]
        pruned = before - len(self._records)
        if pruned:
            logger.info("episodic: pruned %d stale records (older than %d days)", pruned, MAX_EPISODIC_DAYS)
            _safe_write(self._path, {"schema_version": 1, "records": self._records})

    def snapshot(self) -> list[dict[str, Any]]:
        return list(self._records)


def _parse_epoch(iso: str) -> float:
    try:
        return datetime.fromisoformat(iso).timestamp()
    except Exception:  # noqa: BLE001
        return 0.0


# ---------------------------------------------------------------------------
# Semantic memory  (derived from model.json; menu knowledge)
# ---------------------------------------------------------------------------
class SemanticMemory:
    """
    Read-only structured knowledge extracted from model.json.

    Provides menu items, prep statistics, demand thresholds, and feature rules
    without duplicating any data — it always reads the authoritative model.json.
    """

    def __init__(self, model_path: Path = MODEL_FILE) -> None:
        self._path = model_path
        self._bundle: dict[str, Any] = {}
        self._loaded_at: float = 0.0
        self._load()

    def _load(self) -> None:
        raw = _safe_read(self._path, None)
        if raw is not None:
            self._bundle = raw
            self._loaded_at = time.time()
        else:
            logger.warning("semantic: could not load %s", self._path)

    def _ensure_fresh(self) -> None:
        """Reload if model.json has been updated since last load."""
        try:
            mtime = self._path.stat().st_mtime
            if mtime > self._loaded_at:
                self._load()
        except OSError:
            pass

    @property
    def menu_items(self) -> list[str]:
        self._ensure_fresh()
        return sorted(self._bundle.get("models", {}).keys())

    @property
    def services(self) -> list[str]:
        return ["Lunch", "Dinner"]

    @property
    def feature_order(self) -> list[str]:
        self._ensure_fresh()
        return self._bundle.get("feature_order", [])

    @property
    def feature_rules(self) -> dict[str, str]:
        self._ensure_fresh()
        return self._bundle.get("feature_rules", {})

    @property
    def service_thresholds(self) -> dict[str, dict[str, float]]:
        self._ensure_fresh()
        return self._bundle.get("service_thresholds", {})

    @property
    def limitations(self) -> list[str]:
        self._ensure_fresh()
        return self._bundle.get("limitations", [])

    def prep_stats(self, item: str, service: str) -> dict[str, Any]:
        """Return calibration residuals and historical prep stats for an item+service."""
        self._ensure_fresh()
        models = self._bundle.get("models", {})
        if item not in models:
            return {}
        return models[item].get("services", {}).get(service, {})

    def optimization_defaults(self) -> dict[str, Any]:
        self._ensure_fresh()
        return self._bundle.get("optimization_defaults", {"shortage_penalty": 2, "waste_penalty": 1})

    def summary(self) -> dict[str, Any]:
        self._ensure_fresh()
        return {
            "menu_items": self.menu_items,
            "services": self.services,
            "feature_order": self.feature_order,
            "service_thresholds": self.service_thresholds,
            "optimization_defaults": self.optimization_defaults(),
            "limitations": self.limitations,
            "data_range": {
                "start": self._bundle.get("data_start"),
                "end": self._bundle.get("data_end"),
            },
        }


# ---------------------------------------------------------------------------
# Context Manager  (retrieval + prioritisation + size control)
# ---------------------------------------------------------------------------
class ContextManager:
    """
    Assembles a task-scoped context dict from all memory layers.

    Priority (highest → lowest):
        1. Current verified data (semantic: model, thresholds)
        2. Current explicit user instructions (short-term: user_request)
        3. Recent verified operational records (episodic: recent with outcomes)
        4. Older stored memories (long-term config)
        5. Inferences and assumptions — NOT injected automatically

    Stale / irrelevant memories are excluded. Context is capped at
    MAX_CONTEXT_BYTES (soft) to avoid bloating prompts.
    """

    def __init__(
        self,
        short: ShortTermMemory,
        long: LongTermMemory,
        episodic: EpisodicMemory,
        semantic: SemanticMemory,
    ) -> None:
        self.short = short
        self.long = long
        self.episodic = episodic
        self.semantic = semantic

    def retrieve(
        self,
        task: str = "",
        date: str | None = None,
        service: str | None = None,
        max_episodic: int = MAX_CONTEXT_EPISODIC,
    ) -> dict[str, Any]:
        """
        Return a filtered, prioritised context dict for the current task.

        Parameters
        ----------
        task     : free-text task description used for relevance filtering
        date     : specific forecast date (optional)
        service  : "Lunch" or "Dinner" (optional)
        """
        ctx: dict[str, Any] = {}

        # P1 — Semantic (authoritative model knowledge)
        ctx["semantic"] = self.semantic.summary()

        # P2 — Short-term (current user request + active state)
        st = self.short.snapshot()
        if st:
            ctx["current_session"] = st

        # P3 — Episodic (recent predictions with verified outcomes first)
        recent = self.episodic.recent(max_episodic)
        verified = [r for r in recent if r.get("actual_demand") is not None]
        unverified = [r for r in recent if r.get("actual_demand") is None]
        episodic_ctx: list[dict] = []
        if date and service:
            exact = self.episodic.for_date_service(date, service)
            if exact:
                episodic_ctx.append(exact)
        # Add verified first, then unverified, avoid duplicates
        seen_keys: set[tuple] = {(r["date"], r["service"]) for r in episodic_ctx}
        for r in verified + unverified:
            k = (r["date"], r["service"])
            if k not in seen_keys:
                episodic_ctx.append(r)
                seen_keys.add(k)
        avg_mae = self.episodic.average_mae(service)
        ctx["episodic"] = {
            "records": episodic_ctx[:max_episodic],
            "average_mae": avg_mae,
            "note": "Predictions and outcomes stored separately. Outcomes marked None are unverified.",
        }

        # P4 — Long-term (config, preferences, rules)
        lt = self.long.snapshot()
        ctx["long_term"] = {
            "restaurant": lt.get("restaurant", {}),
            "preferences": lt.get("preferences", {}),
            "operational_rules": lt.get("operational_rules", []),
        }

        # Size control — truncate episodic records if over budget
        ctx = _trim_context(ctx)
        return ctx

    def log_event(self, event: str, details: dict[str, Any] | None = None) -> None:
        """Append a named event to the short-term log (not persisted)."""
        events: list = self.short.get("_events", [])
        events.append({"event": event, "details": details or {}, "at": _now_iso()})
        self.short.set("_events", events)


def _trim_context(ctx: dict[str, Any]) -> dict[str, Any]:
    """
    Soft-trim the episodic record list if the full context exceeds MAX_CONTEXT_BYTES.
    Does not truncate semantic or long-term knowledge.
    """
    encoded = json.dumps(ctx, allow_nan=False)
    if len(encoded) <= MAX_CONTEXT_BYTES:
        return ctx
    # Trim episodic records one by one until under budget
    records: list = ctx.get("episodic", {}).get("records", [])
    while len(encoded) > MAX_CONTEXT_BYTES and records:
        records.pop()
        encoded = json.dumps(ctx, allow_nan=False)
    return ctx


# ---------------------------------------------------------------------------
# Top-level agent integration helper
# ---------------------------------------------------------------------------
class RushIQMemory:
    """
    Single entry-point for the RushIQ agent.

    Usage
    -----
    memory = RushIQMemory()

    # Before agent run:
    memory.start_session(user_request="Forecast for 2025-07-01 Dinner")

    # Run the existing RushIQForecast logic, then:
    memory.save_prediction("2025-07-01", "Dinner", predictions, flags)

    # After actual outcomes arrive:
    memory.save_outcome("2025-07-01", "Dinner", actual_demand)

    # Build context for a prompt:
    ctx = memory.context_for("2025-07-01", "Dinner")
    """

    def __init__(self) -> None:
        self.short    = ShortTermMemory()
        self.long     = LongTermMemory()
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()
        self.context  = ContextManager(self.short, self.long, self.episodic, self.semantic)

    # -- session lifecycle ---------------------------------------------------

    def start_session(self, user_request: str = "") -> None:
        self.short.clear()
        self.short.update({
            "user_request": user_request,
            "session_started_at": _now_iso(),
            "state": "active",
        })
        self.context.log_event("session_start", {"user_request": user_request})
        logger.info("memory: session started")

    def end_session(self) -> None:
        self.short.set("state", "complete")
        self.short.set("session_ended_at", _now_iso())
        logger.info("memory: session ended")

    # -- prediction / outcome ------------------------------------------------

    def save_prediction(
        self,
        date: str,
        service: str,
        predictions: dict[str, float],
        flags: dict[str, int],
    ) -> bool:
        ok = self.episodic.record_prediction(date, service, predictions, flags)
        self.context.log_event("prediction_saved", {"date": date, "service": service, "ok": ok})
        return ok

    def save_outcome(
        self,
        date: str,
        service: str,
        actual_demand: dict[str, int],
    ) -> bool:
        ok = self.episodic.record_outcome(date, service, actual_demand)
        self.context.log_event("outcome_saved", {"date": date, "service": service, "ok": ok})
        return ok

    # -- context retrieval ---------------------------------------------------

    def context_for(
        self,
        date: str | None = None,
        service: str | None = None,
        task: str = "",
    ) -> dict[str, Any]:
        ctx = self.context.retrieve(task=task, date=date, service=service)
        self.context.log_event("context_retrieved", {
            "date": date, "service": service, "keys": list(ctx.keys()),
        })
        return ctx

    # -- long-term updates ---------------------------------------------------

    def update_preference(self, key: str, value: Any) -> bool:
        return self.long.update_preference(key, value)

    def add_rule(self, rule: str) -> bool:
        return self.long.add_rule(rule)

    # -- observability -------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Return a brief health / status summary for logging or UI display."""
        return {
            "short_term_keys": list(self.short.snapshot().keys()),
            "long_term_updated_at": self.long.get("updated_at"),
            "episodic_record_count": len(self.episodic.snapshot()),
            "episodic_avg_mae_overall": self.episodic.average_mae(),
            "semantic_menu_items": self.semantic.menu_items,
            "semantic_loaded": bool(self.semantic.menu_items),
            "memory_store_path": str(MEMORY_DIR),
        }
