"""
RushIQ Memory + Context Tests
==============================
Fast, self-contained tests that verify all memory layers and the agent integration.
No external dependencies beyond stdlib + the existing project files.

Run from the repository root:
    python memory/test_memory.py
"""

import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

# Override default memory paths to a temp dir for isolation
_TMP = tempfile.mkdtemp(prefix="rushiq_test_")
os.environ["RUSHIQ_MEMORY_DIR"] = _TMP  # picked up below

# Patch memory paths before importing
import memory_store as ms

ms.MEMORY_DIR = Path(_TMP)
ms.MEMORY_DIR.mkdir(parents=True, exist_ok=True)
ms.LONG_TERM_FILE = ms.MEMORY_DIR / "long_term.json"
ms.EPISODIC_FILE  = ms.MEMORY_DIR / "episodic.json"
ms.MODEL_FILE     = ROOT / "web" / "data" / "model.json"

from memory_store import (
    ShortTermMemory,
    LongTermMemory,
    EpisodicMemory,
    SemanticMemory,
    ContextManager,
    RushIQMemory,
)

# ---------------------------------------------------------------------------
# Minimal test harness
# ---------------------------------------------------------------------------
PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}" + (f" — {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_short_term():
    print("\n[ShortTermMemory]")
    stm = ShortTermMemory()
    stm.set("user_request", "Forecast Dinner 2025-07-01")
    stm.set("active_date", "2025-07-01")
    check("set/get", stm.get("user_request") == "Forecast Dinner 2025-07-01")
    check("default", stm.get("missing_key", "default") == "default")
    stm.update({"state": "active", "foo": 42})
    check("update", stm.get("state") == "active" and stm.get("foo") == 42)
    snap = stm.snapshot()
    check("snapshot is dict", isinstance(snap, dict))
    check("snapshot has keys", "user_request" in snap)
    stm.clear()
    check("clear", stm.get("user_request") is None)


def test_long_term():
    print("\n[LongTermMemory]")
    lt = LongTermMemory(ms.LONG_TERM_FILE)
    check("loaded default", lt.get("restaurant") is not None)
    check("shortage_penalty default", lt.get("preferences", {}).get("shortage_penalty") == 2)
    ok = lt.update_preference("waste_penalty", 3)
    check("update_preference persists", ok and lt.get("preferences", {}).get("waste_penalty") == 3)
    ok2 = lt.add_rule("New test rule.")
    check("add_rule persists", ok2)
    rules = lt.get("operational_rules", [])
    check("rule in list", "New test rule." in rules)
    # Reload from disk
    lt2 = LongTermMemory(ms.LONG_TERM_FILE)
    check("persisted across reload", "New test rule." in lt2.get("operational_rules", []))


def test_episodic():
    print("\n[EpisodicMemory]")
    ep = EpisodicMemory(ms.EPISODIC_FILE)
    preds = {"Brownie": 14.5, "Classic Burger": 32.1, "Cold Coffee": 18.0}
    flags = {"is_holiday": 0, "is_festival_or_event": 0, "is_promotion": 1}

    ok = ep.record_prediction("2025-07-01", "Dinner", preds, flags)
    check("record_prediction returns True", ok)

    # Duplicate should return False
    ok2 = ep.record_prediction("2025-07-01", "Dinner", preds, flags)
    check("duplicate returns False", not ok2)

    rec = ep.for_date_service("2025-07-01", "Dinner")
    check("for_date_service found", rec is not None)
    check("outcome is None before record_outcome", rec["actual_demand"] is None)

    actual = {"Brownie": 13, "Classic Burger": 35, "Cold Coffee": 17}
    ok3 = ep.record_outcome("2025-07-01", "Dinner", actual)
    check("record_outcome returns True", ok3)

    rec2 = ep.for_date_service("2025-07-01", "Dinner")
    check("outcome recorded", rec2["actual_demand"] is not None)
    check("mae computed", rec2["mae"] is not None and rec2["mae"] >= 0)
    check("bias computed", rec2["bias"] is not None)

    # Missing outcome
    ok_miss = ep.record_outcome("1999-01-01", "Lunch", {"X": 1})
    check("missing prediction returns False", not ok_miss)

    recent = ep.recent(5)
    check("recent returns list", isinstance(recent, list) and len(recent) >= 1)

    # Reload from disk
    ep2 = EpisodicMemory(ms.EPISODIC_FILE)
    check("persisted across reload", ep2.for_date_service("2025-07-01", "Dinner") is not None)

    avg = ep2.average_mae("Dinner")
    check("average_mae returns float", avg is not None and avg >= 0)


def test_semantic():
    print("\n[SemanticMemory]")
    model_path = ROOT / "web" / "data" / "model.json"
    if not model_path.exists():
        print("  SKIP  model.json not found — run python training/train.py first")
        return
    sem = SemanticMemory(model_path)
    items = sem.menu_items
    check("menu_items non-empty", len(items) > 0)
    check("Brownie in menu", "Brownie" in items)
    thresholds = sem.service_thresholds
    check("thresholds has Lunch", "Lunch" in thresholds)
    check("thresholds has Dinner", "Dinner" in thresholds)
    rules = sem.feature_rules
    check("feature_rules non-empty", bool(rules))
    prep = sem.prep_stats("Brownie", "Lunch")
    check("prep_stats has residuals", "residuals" in prep)
    summary = sem.summary()
    check("summary is dict", isinstance(summary, dict))
    check("summary has menu_items", "menu_items" in summary)


def test_context_manager():
    print("\n[ContextManager]")
    # Fresh isolated instances
    short  = ShortTermMemory()
    long   = LongTermMemory(ms.MEMORY_DIR / "lt_ctx.json")
    ep     = EpisodicMemory(ms.MEMORY_DIR / "ep_ctx.json")
    sem    = SemanticMemory(ROOT / "web" / "data" / "model.json")
    ctx_mgr = ContextManager(short, long, ep, sem)

    short.set("user_request", "7-day forecast")
    short.set("active_date", "2025-07-01")

    # Insert an episodic record
    ep.record_prediction("2025-07-01", "Dinner",
                         {"Brownie": 15.0}, {"is_holiday": 0, "is_festival_or_event": 0, "is_promotion": 0})

    ctx = ctx_mgr.retrieve(task="Forecast", date="2025-07-01", service="Dinner")
    check("context is dict", isinstance(ctx, dict))
    check("semantic in context", "semantic" in ctx)
    check("long_term in context", "long_term" in ctx)
    check("episodic in context", "episodic" in ctx)
    check("current_session in context", "current_session" in ctx)
    check("episodic records list", isinstance(ctx["episodic"]["records"], list))
    # Empty episodic does not break
    ep2 = EpisodicMemory(ms.MEMORY_DIR / "ep_empty.json")
    ctx_mgr2 = ContextManager(short, long, ep2, sem)
    ctx2 = ctx_mgr2.retrieve()
    check("empty episodic is safe", ctx2["episodic"]["records"] == [])


def test_rushiq_memory_integration():
    print("\n[RushIQMemory integration]")
    # Redirect paths
    ms.LONG_TERM_FILE = ms.MEMORY_DIR / "lt_int.json"
    ms.EPISODIC_FILE  = ms.MEMORY_DIR / "ep_int.json"

    mem = RushIQMemory()
    mem.start_session("Full forecast run")
    check("short-term has user_request", mem.short.get("user_request") == "Full forecast run")

    preds = {"Brownie": 12.0, "Classic Burger": 28.5}
    flags = {"is_holiday": 0, "is_festival_or_event": 0, "is_promotion": 0}
    ok = mem.save_prediction("2025-07-02", "Lunch", preds, flags)
    check("save_prediction ok", ok)

    ctx = mem.context_for("2025-07-02", "Lunch")
    check("context_for returns dict", isinstance(ctx, dict))

    ok2 = mem.save_outcome("2025-07-02", "Lunch", {"Brownie": 11, "Classic Burger": 30})
    check("save_outcome ok", ok2)

    mem.end_session()
    check("state complete", mem.short.get("state") == "complete")

    status = mem.status()
    check("status dict", isinstance(status, dict))
    check("status has episodic_record_count", "episodic_record_count" in status)
    check("status has semantic_menu_items", "semantic_menu_items" in status)


def test_agent():
    print("\n[RushIQAgent]")
    model_path = ROOT / "web" / "data" / "model.json"
    if not model_path.exists():
        print("  SKIP  model.json not found")
        return

    # Import here so memory paths are already patched
    from agent import RushIQAgent
    ms.LONG_TERM_FILE = ms.MEMORY_DIR / "lt_agent.json"
    ms.EPISODIC_FILE  = ms.MEMORY_DIR / "ep_agent.json"

    agent = RushIQAgent(model_path)
    flags = {"is_holiday": 0, "is_festival_or_event": 0, "is_promotion": 0}
    result = agent.run("2025-07-01", "Dinner", flags, user_request="Dinner forecast July 1")

    check("result is dict", isinstance(result, dict))
    check("forecast present", result.get("forecast") is not None)
    check("no errors", result["errors"] == [], str(result["errors"]))
    fc = result["forecast"]
    check("forecast has items", isinstance(fc.get("items"), list) and len(fc["items"]) > 0)
    check("forecast has total_units", isinstance(fc.get("total_units"), float))
    check("forecast has demand_class", fc.get("demand_class") in ("Low", "Normal", "High"))
    check("context_summary present", isinstance(result.get("context_summary"), dict))
    check("memory_status present", isinstance(result.get("memory_status"), dict))
    check("episodic_record_count >= 1", result["memory_status"].get("episodic_record_count", 0) >= 1)

    # Empty episodic does not break a second call
    result2 = agent.run("2025-07-02", "Lunch", flags)
    check("second call works", result2.get("forecast") is not None)
    check("no errors on second call", result2["errors"] == [], str(result2["errors"]))

    # record_actual_outcome
    actual = {it["menu_item"]: int(it["predicted_units"]) for it in fc["items"]}
    ok = agent.record_actual_outcome("2025-07-01", "Dinner", actual)
    check("record_actual_outcome ok", ok)


def test_js_context_shim():
    """Verify the JS context shim file is valid JSON-embeddable."""
    print("\n[JS context shim]")
    shim = ROOT / "web" / "js" / "memory_context.js"
    check("shim file exists", shim.exists(), str(shim))
    if shim.exists():
        content = shim.read_text(encoding="utf-8")
        check("shim non-empty", len(content) > 100)
        check("RushIQMemory defined", "RushIQMemory" in content)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("RushIQ Memory Tests")
    print("=" * 60)

    for fn in [
        test_short_term,
        test_long_term,
        test_episodic,
        test_semantic,
        test_context_manager,
        test_rushiq_memory_integration,
        test_agent,
        test_js_context_shim,
    ]:
        try:
            fn()
        except Exception:
            global FAIL
            FAIL += 1
            print(f"  ERROR in {fn.__name__}:")
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"Results: {PASS} passed, {FAIL} failed")
    print(f"{'=' * 60}")
    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()
