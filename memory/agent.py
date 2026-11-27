"""
RushIQ Agent
============
Thin orchestration wrapper that integrates the existing RushIQForecast browser
library's pure Python equivalent (via model.json) with the memory layer.

This preserves the existing forecast.js contract completely. The agent is an
additive layer; it does not replace or modify forecast.js or train.py.

Entry points
------------
agent = RushIQAgent()
result = agent.run(date="2025-07-01", service="Dinner", flags={...})
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

from memory_store import RushIQMemory

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = ROOT / "web" / "data" / "model.json"

FEATURES = [
    "is_dinner",
    "weekday_1", "weekday_2", "weekday_3", "weekday_4", "weekday_5", "weekday_6",
    "is_holiday", "is_festival_or_event", "is_promotion",
]


# ---------------------------------------------------------------------------
# Pure-Python mirror of forecast.js  (no browser dependency)
# ---------------------------------------------------------------------------
def _build_features(bundle: dict, date: str, service: str, flags: dict[str, int]) -> list[float]:
    from datetime import datetime, timezone
    ts = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    weekday = (ts.weekday())  # Monday=0 … Sunday=6
    values: dict[str, int] = {"is_dinner": int(service == "Dinner")}
    for day in range(1, 7):
        values[f"weekday_{day}"] = int(weekday == day)
    for name in ("is_holiday", "is_festival_or_event", "is_promotion"):
        if flags.get(name) not in (0, 1):
            raise ValueError(f"Supply an explicit 0 or 1 for {name}")
        values[name] = flags[name]
    feature_order = bundle.get("feature_order", FEATURES)
    return [float(values[f]) for f in feature_order]


def _predict_item(bundle: dict, item: str, feature_values: list[float]) -> dict[str, Any]:
    model = bundle["models"][item]
    intercept = model["intercept"]
    coefficients = model["coefficients"]
    raw = intercept + sum(c * v for c, v in zip(coefficients, feature_values))
    return {
        "menu_item": item,
        "predicted_units": max(0.0, raw),
        "intercept": intercept,
        "raw_prediction": raw,
    }


def _forecast_service(bundle: dict, date: str, service: str, flags: dict[str, int]) -> dict[str, Any]:
    feature_values = _build_features(bundle, date, service, flags)
    items = [_predict_item(bundle, item, feature_values) for item in sorted(bundle["models"])]
    total = sum(it["predicted_units"] for it in items)
    thresholds = bundle["service_thresholds"][service]
    demand_class = "Normal"
    if total < thresholds["low_below"]:
        demand_class = "Low"
    elif total > thresholds["high_above"]:
        demand_class = "High"
    return {
        "date": date,
        "time_period": service,
        "items": items,
        "total_units": total,
        "demand_class": demand_class,
        "spike_alert": total > thresholds["spike_above"],
    }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class RushIQAgent:
    """
    Wraps the forecast engine with memory-aware context management.

    The agent:
    1. Retrieves relevant context from memory.
    2. Runs the deterministic Ridge forecast (unchanged logic from train.py/forecast.js).
    3. Saves the prediction to episodic memory.
    4. Returns the forecast result with attached context metadata.

    Memory failures never block the forecast.
    """

    def __init__(self, model_path: Path = MODEL_FILE) -> None:
        self.memory = RushIQMemory()
        self._model_path = model_path
        self._bundle: dict[str, Any] | None = None

    def _load_bundle(self) -> dict[str, Any]:
        if self._bundle is None:
            raw = self._model_path.read_text(encoding="utf-8")
            self._bundle = json.loads(raw)
        return self._bundle

    def run(
        self,
        date: str,
        service: str,
        flags: dict[str, int],
        user_request: str = "",
        save_prediction: bool = True,
    ) -> dict[str, Any]:
        """
        Run a forecast for a single date+service, with memory integration.

        Parameters
        ----------
        date           : YYYY-MM-DD
        service        : "Lunch" or "Dinner"
        flags          : {"is_holiday": 0|1, "is_festival_or_event": 0|1, "is_promotion": 0|1}
        user_request   : free-text request (stored in short-term memory)
        save_prediction: whether to persist the prediction in episodic memory

        Returns
        -------
        dict with keys: forecast, context_summary, memory_status, errors
        """
        errors: list[str] = []

        # -- session start ---------------------------------------------------
        try:
            self.memory.start_session(user_request=user_request or f"Forecast {date} {service}")
            self.memory.short.set("active_date", date)
            self.memory.short.set("active_service", service)
            self.memory.short.set("active_flags", flags)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"memory.start_session failed: {exc}")
            logger.warning("agent: memory start_session error: %s", exc)

        # -- context retrieval -----------------------------------------------
        context: dict[str, Any] = {}
        try:
            context = self.memory.context_for(date=date, service=service, task=user_request)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"memory.context_for failed: {exc}")
            logger.warning("agent: context retrieval error: %s", exc)

        # -- forecast (core, must not fail) ----------------------------------
        try:
            bundle = self._load_bundle()
            forecast = _forecast_service(bundle, date, service, flags)
        except Exception as exc:
            errors.append(f"forecast failed: {exc}")
            logger.error("agent: forecast error: %s", exc)
            self.memory.end_session()
            return {"forecast": None, "context_summary": {}, "memory_status": {}, "errors": errors}

        # -- save prediction to episodic memory ------------------------------
        if save_prediction:
            try:
                predictions = {it["menu_item"]: it["predicted_units"] for it in forecast["items"]}
                self.memory.save_prediction(date, service, predictions, flags)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"memory.save_prediction failed: {exc}")
                logger.warning("agent: save_prediction error: %s", exc)

        # -- context summary (non-sensitive subset for response) -------------
        context_summary = {
            "episodic_records_used": len(context.get("episodic", {}).get("records", [])),
            "avg_mae_this_service": context.get("episodic", {}).get("average_mae"),
            "menu_items_known": len(context.get("semantic", {}).get("menu_items", [])),
            "operational_rules": context.get("long_term", {}).get("operational_rules", []),
        }

        # -- memory status for observability ---------------------------------
        memory_status: dict[str, Any] = {}
        try:
            memory_status = self.memory.status()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"memory.status failed: {exc}")

        self.memory.short.set("state", "forecast_complete")
        self.memory.end_session()

        return {
            "forecast": forecast,
            "context_summary": context_summary,
            "memory_status": memory_status,
            "errors": errors,
        }

    def record_actual_outcome(
        self,
        date: str,
        service: str,
        actual_demand: dict[str, int],
    ) -> bool:
        """
        After actual sales are known, update the episodic record.
        Call this separately; never conflate predicted with actual.
        """
        try:
            return self.memory.save_outcome(date, service, actual_demand)
        except Exception as exc:  # noqa: BLE001
            logger.warning("agent: record_actual_outcome error: %s", exc)
            return False
