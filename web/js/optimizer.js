(function (global) {
  "use strict";

  function evaluateQuantity(scenarios, quantity, weights) {
    let shortCount = 0;
    let wasteCount = 0;
    let shortage = 0;
    let waste = 0;
    for (const demand of scenarios) {
      shortCount += Number(demand > quantity);
      wasteCount += Number(demand < quantity);
      shortage += Math.max(0, demand - quantity);
      waste += Math.max(0, quantity - demand);
    }
    return {
      shortageRisk: shortCount / scenarios.length,
      wasteRisk: wasteCount / scenarios.length,
      expectedShortage: shortage / scenarios.length,
      expectedWaste: waste / scenarios.length,
      loss: (weights.shortage_penalty * shortage + weights.waste_penalty * waste) / scenarios.length
    };
  }

  function recommend(prediction, history, weights) {
    const scenarios = history.residuals.map(residual => Math.max(0, prediction + residual));
    scenarios.sort((a, b) => a - b);
    const target = weights.shortage_penalty / (weights.shortage_penalty + weights.waste_penalty);
    const pivot = scenarios[Math.ceil(target * scenarios.length) - 1];
    // The continuous minimum is at the weighted quantile. Check both adjacent integers.
    const below = Math.floor(pivot);
    const above = Math.ceil(pivot);
    const lower = evaluateQuantity(scenarios, below, weights);
    const upper = evaluateQuantity(scenarios, above, weights);
    const best = upper.loss < lower.loss - 1e-9
      ? { quantity: above, ...upper }
      : { quantity: below, ...lower };
    return {
      ...best,
      typicalQuantity: history.typical_prepared,
      typicalRisk: evaluateQuantity(scenarios, history.typical_prepared, weights).shortageRisk,
      sampleSize: scenarios.length
    };
  }

  global.RushIQOptimizer = Object.freeze({ recommend });
})(globalThis);
