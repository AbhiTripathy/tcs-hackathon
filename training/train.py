"""Train reproducible calendar-based demand models and export browser assets."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parents[1]
KEY = ["date", "time_period", "menu_item"]
FLAGS = ["is_holiday", "is_festival_or_event", "is_promotion"]
FEATURES = ["is_dinner"] + [f"weekday_{i}" for i in range(1, 7)] + FLAGS
TRAIN_END = "2025-04-30"
CALIBRATION_END = "2025-05-30"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_data(directory):
    orders = pd.read_csv(directory / "orders.csv")
    prep = pd.read_csv(directory / "menu_preparation.csv")
    calendar = pd.read_csv(directory / "calendar.csv")
    required = [KEY + ["order_channel", "quantity_ordered"],
                KEY + ["quantity_prepared", "quantity_sold", "unsold_quantity",
                       "shortage_quantity"], ["date"] + FLAGS]
    for frame, columns in zip([orders, prep, calendar], required):
        require(set(columns).issubset(frame.columns), f"Missing required columns: {columns}")
        require(not frame[columns].isna().any().any(), "Missing required values")
        frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="raise")
    require(not orders.duplicated(KEY + ["order_channel"]).any(), "Duplicate order keys")
    require(not prep.duplicated(KEY).any(), "Duplicate preparation keys")
    require(not calendar.duplicated("date").any(), "Duplicate calendar dates")
    for frame, columns in [(orders, ["quantity_ordered"]), (prep, required[1][3:])]:
        values = frame[columns].to_numpy(dtype=float)
        require(bool(np.isfinite(values).all() and (values >= 0).all()
                     and (values == np.floor(values)).all()), "Counts must be finite nonnegative integers")
    require(calendar[FLAGS].isin([0, 1]).all().all(), "Calendar flags must be binary")
    demand = orders.groupby(KEY, as_index=False)["quantity_ordered"].sum()
    data = demand.merge(prep[required[1]], on=KEY, how="outer", validate="one_to_one")
    require(not data.isna().any().any(), "Orders and preparation keys differ")
    require(data.quantity_ordered.eq(data.quantity_sold).all(), "Orders do not reconcile to sold units")
    require(data.unsold_quantity.eq((data.quantity_prepared - data.quantity_sold).clip(lower=0)).all(),
            "Unsold quantities do not reconcile")
    require(data.shortage_quantity.eq((data.quantity_sold - data.quantity_prepared).clip(lower=0)).all(),
            "Shortage quantities do not reconcile")
    data = data.merge(calendar[["date"] + FLAGS], on="date", how="left", validate="many_to_one")
    require(not data[FLAGS].isna().any().any(), "Missing calendar coverage")
    require(set(data.time_period) == {"Lunch", "Dinner"}, "Expected Lunch and Dinner services")
    dates = pd.date_range(data.date.min(), data.date.max())
    require(len(data) == len(dates) * 2 * data.menu_item.nunique(), "Incomplete item/service grid")
    data["weekday"] = data.date.dt.day_of_week
    return data.sort_values(KEY).reset_index(drop=True)


def features(data):
    matrix = pd.DataFrame(index=data.index)
    matrix["is_dinner"] = data.time_period.eq("Dinner").astype(int)
    for weekday in range(1, 7):
        matrix[f"weekday_{weekday}"] = data.weekday.eq(weekday).astype(int)
    for flag in FLAGS:
        matrix[flag] = data[flag].astype(int)
    return matrix[FEATURES].to_numpy(dtype=float)


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    error = np.abs(actual - predicted)
    denominator = float(actual.sum())
    return {"mae_units": float(error.mean()),
            "wape": float(error.sum() / denominator) if denominator else None,
            "bias_units": float((predicted - actual).mean()), "rows": len(actual)}


def train(directory, output):
    data = load_data(directory)
    train_data = data[data.date <= TRAIN_END]
    calibration = data[(data.date > TRAIN_END) & (data.date <= CALIBRATION_END)]
    test = data[data.date > CALIBRATION_END].copy()
    require(all(len(part) for part in [train_data, calibration, test]), "Empty chronological partition")
    test["prediction"] = 0.0
    test["baseline"] = 0.0
    models = {}
    parity_cases = []
    for item in sorted(data.menu_item.unique()):
        fit = train_data[train_data.menu_item.eq(item)]
        cal = calibration[calibration.menu_item.eq(item)]
        heldout = test[test.menu_item.eq(item)]
        model = Ridge(alpha=1.0, solver="svd").fit(features(fit), fit.quantity_ordered)
        predictions = np.maximum(0, model.predict(features(heldout)))
        test.loc[heldout.index, "prediction"] = predictions
        baseline = fit.groupby(["time_period", "weekday"]).quantity_ordered.mean()
        test.loc[heldout.index, "baseline"] = [float(baseline.loc[(r.time_period, r.weekday)])
                                                for r in heldout.itertuples()]
        cal_prediction = np.maximum(0, model.predict(features(cal)))
        residual = cal.quantity_ordered.to_numpy() - cal_prediction
        service_data = {}
        for service in ["Lunch", "Dinner"]:
            historical = fit[fit.time_period.eq(service)]
            service_data[service] = {
                "residuals": residual[cal.time_period.eq(service)].tolist(),
                "typical_prepared": float(historical.quantity_prepared.median()),
                "historical_prepared_units": int(historical.quantity_prepared.sum()),
                "historical_unsold_units": int(historical.unsold_quantity.sum()),
                "historical_waste_rate": float(historical.unsold_quantity.sum() /
                                               historical.quantity_prepared.sum()),
                "weekday_mean_units": {str(k): float(v) for k, v in
                                       historical.groupby("weekday").quantity_ordered.mean().items()},
            }
        models[item] = {"intercept": float(model.intercept_), "coefficients": model.coef_.tolist(),
                        "services": service_data,
                        "test_metrics": metrics(heldout.quantity_ordered, predictions)}
        for index in [0, len(heldout) - 1]:
            parity_cases.append({"menu_item": item, "features": features(heldout)[index].tolist(),
                                 "expected_prediction": float(predictions[index])})
    totals = train_data.groupby(["date", "time_period"]).quantity_ordered.sum().reset_index()
    thresholds = {}
    for service, rows in totals.groupby("time_period"):
        low, high, spike = np.quantile(rows.quantity_ordered, [0.25, 0.75, 0.9])
        thresholds[service] = {"low_below": float(low), "high_above": float(high),
                               "spike_above": float(spike)}
    report = {"item_rows": {"model": metrics(test.quantity_ordered, test.prediction),
                             "baseline": metrics(test.quantity_ordered, test.baseline)}}
    service_test = test.groupby(["date", "time_period"])[["quantity_ordered", "prediction", "baseline"]].sum()
    report["service_totals"] = {"model": metrics(service_test.quantity_ordered, service_test.prediction),
                                 "baseline": metrics(service_test.quantity_ordered, service_test.baseline)}
    fingerprints = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                    for name in ["orders.csv", "menu_preparation.csv", "calendar.csv"]}
    bundle = {
        "schema_version": 1, "model_type": "ridge", "alpha": 1.0,
        "sklearn_version": sklearn.__version__, "feature_order": FEATURES,
        "feature_rules": {"weekday": "Monday=0 through Sunday=6; Monday is reference",
                          "is_dinner": "Dinner=1, Lunch=0", "flags": "Explicit known scenario inputs, 0 or 1"},
        "inference": "max(0, intercept + sum(coefficients[j] * features[j]))",
        "classification": "Low if total < low_below, High if total > high_above, otherwise Normal",
        "horizon_days": 7, "data_start": str(data.date.min().date()),
        "data_end": str(data.date.max().date()),
        "split": {"train_end": TRAIN_END, "calibration_end": CALIBRATION_END,
                  "test_start": str(test.date.min().date()), "test_end": str(test.date.max().date())},
        "source_sha256": fingerprints, "models": models, "service_thresholds": thresholds,
        "optimization_defaults": {"shortage_penalty": 2, "waste_penalty": 1},
        "uncertainty_method": "Empirical additive calibration residuals by item/service, clipped at zero",
        "limitations": ["Synthetic item units proxy for demand; unsold units proxy for waste",
                        "Risk estimates use 30 calibration residuals per item/service",
                        "Calendar effects have few observations; future flags must be supplied",
                        "Forecasts beyond the historical range assume stable weekly demand patterns"],
        "evaluation": report, "parity_cases": parity_cases,
    }
    records = []
    for row in test.itertuples():
        records.append({"date": str(row.date.date()), "time_period": row.time_period,
                        "menu_item": row.menu_item, "features": features(test.loc[[row.Index]])[0].tolist(),
                        "actual_demand": int(row.quantity_ordered),
                        "historical_prepared": int(row.quantity_prepared),
                        "historical_unsold": int(row.unsold_quantity),
                        "historical_shortage": int(row.shortage_quantity),
                        "expected_prediction": float(row.prediction), "baseline_prediction": float(row.baseline)})
    replay = {"schema_version": 1, "evaluation": report,
              "protocol": "Forecast and optimize using features and model.json before reading actual outcomes",
              "records": records}
    output.mkdir(parents=True, exist_ok=True)
    for name, artifact in [("model.json", bundle), ("backtest.json", replay)]:
        encoded = json.dumps(artifact, indent=2, allow_nan=False) + "\n"
        temporary = output / (name + ".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(output / name)
    print(json.dumps(report, indent=2))
    print(f"Exported {len(models)} trained models and {len(records)} held-out records to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "Datasets")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "web" / "data")
    arguments = parser.parse_args()
    train(arguments.data_dir, arguments.output_dir)
