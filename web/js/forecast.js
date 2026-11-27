/* Load as a classic script. All inference is local and deterministic. */
(function (global) {
  "use strict";

  function buildFeatures(bundle, date, service, flags) {
    if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) {
      throw new Error("Expected a date in YYYY-MM-DD format");
    }
    const timestamp = new Date(date + "T00:00:00Z");
    if (!Number.isFinite(timestamp.getTime()) || timestamp.toISOString().slice(0, 10) !== date) {
      throw new Error("Invalid forecast date");
    }
    if (!["Lunch", "Dinner"].includes(service)) {
      throw new Error("Expected Lunch or Dinner");
    }
    const values = { is_dinner: Number(service === "Dinner") };
    const weekday = (timestamp.getUTCDay() + 6) % 7;
    for (let day = 1; day <= 6; day += 1) {
      values["weekday_" + day] = Number(weekday === day);
    }
    for (const name of ["is_holiday", "is_festival_or_event", "is_promotion"]) {
      if (!flags || ![0, 1].includes(flags[name])) {
        throw new Error("Supply an explicit 0 or 1 for " + name);
      }
      values[name] = flags[name];
    }
    return bundle.feature_order.map(function (name) {
      if (!Object.hasOwn(values, name)) throw new Error("Unknown feature " + name);
      return values[name];
    });
  }

  function predictItem(bundle, item, featureValues) {
    const model = bundle.models[item];
    if (!model) throw new Error("Unknown menu item " + item);
    if (featureValues.length !== model.coefficients.length || !featureValues.every(Number.isFinite)) {
      throw new Error("Invalid feature vector");
    }
    const contributions = model.coefficients.map(function (coefficient, index) {
      return { feature: bundle.feature_order[index], value: coefficient * featureValues[index] };
    });
    const rawPrediction = contributions.reduce(function (total, contribution) {
      return total + contribution.value;
    }, model.intercept);
    return { menu_item: item, predicted_units: Math.max(0, rawPrediction),
      intercept: model.intercept, raw_prediction: rawPrediction, contributions: contributions };
  }

  function forecastService(bundle, date, service, flags) {
    const values = buildFeatures(bundle, date, service, flags);
    const items = Object.keys(bundle.models).map(function (item) {
      return predictItem(bundle, item, values);
    });
    const total = items.reduce(function (sum, item) { return sum + item.predicted_units; }, 0);
    const thresholds = bundle.service_thresholds[service];
    let demandClass = "Normal";
    if (total < thresholds.low_below) demandClass = "Low";
    if (total > thresholds.high_above) demandClass = "High";
    return { date: date, time_period: service, items: items, total_units: total,
      demand_class: demandClass, spike_alert: total > thresholds.spike_above };
  }

  global.RushIQForecast = Object.freeze({ buildFeatures: buildFeatures,
    predictItem: predictItem, forecastService: forecastService });
})(globalThis);
