(function (global) {
  "use strict";

  function replay(bundle, records) {
    return records.map(row => {
      const forecast = global.RushIQForecast.predictItem(bundle, row.menu_item, row.features);
      const history = bundle.models[row.menu_item].services[row.time_period];
      const plan = global.RushIQOptimizer.recommend(forecast.predicted_units, history, bundle.optimization_defaults);
      return {
        ...row,
        prediction: forecast.predicted_units,
        recommended: plan.quantity,
        simulated_shortage: Math.max(0, row.actual_demand - plan.quantity),
        simulated_waste: Math.max(0, plan.quantity - row.actual_demand)
      };
    });
  }

  function summarize(rows, weights) {
    const result = { actual: 0, error: 0, baselineError: 0, historicalWaste: 0,
      simulatedWaste: 0, historicalShortage: 0, simulatedShortage: 0 };
    for (const row of rows) {
      result.actual += row.actual_demand;
      result.error += Math.abs(row.prediction - row.actual_demand);
      result.baselineError += Math.abs(row.baseline_prediction - row.actual_demand);
      result.historicalWaste += row.historical_unsold;
      result.simulatedWaste += row.simulated_waste;
      result.historicalShortage += row.historical_shortage;
      result.simulatedShortage += row.simulated_shortage;
    }
    const loss = (shortage, waste) => shortage * weights.shortage_penalty + waste * weights.waste_penalty;
    return {
      ...result,
      wape: result.actual ? result.error / result.actual : null,
      baselineWape: result.actual ? result.baselineError / result.actual : null,
      historicalLoss: loss(result.historicalShortage, result.historicalWaste),
      simulatedLoss: loss(result.simulatedShortage, result.simulatedWaste)
    };
  }

  global.RushIQBacktest = Object.freeze({ replay, summarize });
})(globalThis);
