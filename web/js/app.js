(() => {
  "use strict";

  const byId = id => document.getElementById(id);
  const forecast = globalThis.RushIQForecast;
  const optimizer = globalThis.RushIQOptimizer;
  const backtest = globalThis.RushIQBacktest;
  const number = new Intl.NumberFormat("en", { maximumFractionDigits: 0 });
  const percent = new Intl.NumberFormat("en", { style: "percent", maximumFractionDigits: 1 });
  const readableDate = new Intl.DateTimeFormat("en", { weekday: "long", month: "short", day: "numeric", timeZone: "UTC" });
  const weekday = new Intl.DateTimeFormat("en", { weekday: "short", timeZone: "UTC" });
  const rangeDate = new Intl.DateTimeFormat("en", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" });
  const toDate = date => new Date(date + "T00:00:00Z");
  const flagDefaults = { is_holiday: 0, is_festival_or_event: 0, is_promotion: 0 };
  const featureNames = {
    is_dinner: "Dinner", is_holiday: "Holiday", is_festival_or_event: "Event", is_promotion: "Promotion"
  };
  for (const [index, name] of ["Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].entries()) {
    featureNames["weekday_" + (index + 1)] = name;
  }
  const pages = {
    forecast: { title: "Demand overview", description: "Plan the next seven days of lunch and dinner service." },
    kitchen: { title: "Kitchen plan", description: "Review prep quantities, risks, and past leftovers for each item." },
    backtest: { title: "Backtest", description: "Compare forecasts and prep recommendations with held-out history." }
  };
  const flagsByDate = new Map();
  const shortageThreshold = 0.5;
  let bundle;
  let replayRows;
  let dates;
  let selectedDate;
  let selectedService = "Lunch";

  function safe(value) {
    const characters = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
    return String(value).replace(/[&<>"']/g, character => characters[character]);
  }

  async function readJson(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Could not load ${url} (HTTP ${response.status})`);
    return response.json();
  }

  function upcomingDates(today, count) {
    return Array.from({ length: count }, (_, index) =>
      new Date(Date.UTC(today.getFullYear(), today.getMonth(), today.getDate() + index + 1))
        .toISOString().slice(0, 10));
  }

  function metric(label, value, context, icon, featured = false) {
    return `<article class="glass metric${featured ? " featured" : ""}">
      <span class="metric-icon" aria-hidden="true">${safe(icon)}</span>
      <div><span class="metric-label">${safe(label)}</span><strong class="metric-value">${safe(value)}</strong><small>${safe(context)}</small></div>
    </article>`;
  }

  function renderWeek(week) {
    const activeButton = document.activeElement?.closest("[data-date]");
    const step = Math.max(50, Math.ceil(Math.max(...week.map(day => day.total_units)) / 150) * 50);
    const ticks = [step * 3, step * 2, step, 0].map((value, index) =>
      `<span style="top:${index * 100 / 3}%">${number.format(value)}</span>`).join("");
    const bars = week.map(day => {
      const date = toDate(day.date);
      const units = number.format(day.total_units);
      return `<button type="button" class="bar-button" data-date="${day.date}" aria-pressed="${day.date === selectedDate}"
        aria-label="${safe(readableDate.format(date))}, ${units} forecast units, ${day.demand_class} demand">
        <span class="bar-track" aria-hidden="true"><span class="bar-fill" style="--height:${day.total_units / (step * 3) * 100}%"><span class="bar-value">${units}</span></span></span>
        <span class="bar-day" aria-hidden="true">${weekday.format(date)}</span>
        <span class="bar-date" aria-hidden="true">${date.getUTCDate()}</span>
      </button>`;
    }).join("");
    byId("weekly-chart").innerHTML = `<div class="week-plot"><div class="axis" aria-hidden="true">${ticks}</div><div class="bars">${bars}</div></div>`;
    if (activeButton) byId("weekly-chart").querySelector(`[data-date="${activeButton.dataset.date}"]`)?.focus({ preventScroll: true });
  }

  function renderAlerts(service, shortages, label) {
    const alerts = [];
    if (service.spike_alert) {
      alerts.push({
        icon: "↑",
        title: "Demand spike",
        message: `${number.format(service.total_units)} units, above the historical spike threshold for ${service.time_period.toLowerCase()}.`
      });
    }
    for (const item of shortages) {
      alerts.push({
        icon: "!",
        title: `${item.menu_item} shortage risk`,
        message: `${percent.format(item.plan.typicalRisk)} at typical prep of ${number.format(item.plan.typicalQuantity)} units. Plan ${number.format(item.plan.quantity)}.`
      });
    }
    byId("alert-date").textContent = label;
    byId("alert-count").textContent = `${alerts.length} ${alerts.length === 1 ? "alert" : "alerts"}`;
    byId("alert-list").innerHTML = (alerts.length ? alerts : [{
      icon: "✓", title: "No alerts for this service",
      message: "Demand is at or below the spike threshold and each item has less than 50% shortage risk at typical prep."
    }]).map(alert => `<li><span class="alert-symbol" aria-hidden="true">${safe(alert.icon)}</span><div><strong>${safe(alert.title)}</strong><small>${safe(alert.message)}</small></div></li>`).join("");
  }

  function riskCell(risk) {
    return `${percent.format(risk)} <span class="risk-mark" style="--risk:${risk * 100}%" aria-hidden="true"></span>`;
  }

  function renderPrep(items, label) {
    byId("prep-period").textContent = label;
    byId("prep-rows").innerHTML = items.map(item => {
      const drivers = item.contributions.filter(driver => Math.abs(driver.value) >= 0.5)
        .sort((a, b) => Math.abs(b.value) - Math.abs(a.value)).slice(0, 3);
      const effects = drivers.map(driver => `${featureNames[driver.feature]} ${driver.value > 0 ? "+" : "−"}${number.format(Math.abs(driver.value))}`);
      const forecastReason = effects.length
        ? `Largest forecast adjustments from the model starting point: ${effects.join(", ")} units.`
        : "The forecast is close to the model starting point.";
      const difference = item.plan.quantity - Math.round(item.predicted_units);
      const prepReason = difference === 0
        ? "The 2:1 penalty and past errors yield the same prep quantity as the rounded forecast."
        : `The 2:1 penalty and past errors suggest ${number.format(Math.abs(difference))} ${difference > 0 ? "more" : "fewer"} units than the rounded forecast.`;
      return `<tr><td class="item-name">${safe(item.menu_item)}</td><td>${number.format(item.predicted_units)}</td>
        <td class="prep-quantity">${number.format(item.plan.quantity)}</td>
        <td>${riskCell(item.plan.shortageRisk)}</td><td>${riskCell(item.plan.wasteRisk)}</td>
        <td><details class="item-reason"><summary aria-label="Explain ${safe(item.menu_item)}">Details</summary><p>${safe(forecastReason)} ${safe(prepReason)}</p></details></td></tr>`;
    }).join("");
  }

  function renderWaste(items) {
    byId("waste-description").textContent = `${selectedService} · Training period, ranked by unsold units`;
    byId("waste-rows").innerHTML = [...items].sort((a, b) => b.history.historical_unsold_units - a.history.historical_unsold_units)
      .map(item => `<tr><td class="item-name">${safe(item.menu_item)}</td><td>${number.format(item.history.historical_unsold_units)}</td><td>${riskCell(item.history.historical_waste_rate)}</td></tr>`).join("");
  }

  function renderPlanning() {
    const week = dates.map(date => forecast.forecastService(bundle, date, selectedService, flagsByDate.get(date)));
    const service = week.find(day => day.date === selectedDate);
    const items = service.items.map(item => {
      const history = bundle.models[item.menu_item].services[selectedService];
      return { ...item, history, plan: optimizer.recommend(item.predicted_units, history, bundle.optimization_defaults) };
    });
    const shortages = items.filter(item => item.plan.typicalRisk >= shortageThreshold)
      .sort((a, b) => b.plan.typicalRisk - a.plan.typicalRisk);
    const label = `${readableDate.format(toDate(selectedDate))} · ${selectedService}`;
    const totalPrep = items.reduce((sum, item) => sum + item.plan.quantity, 0);
    byId("metrics").innerHTML = [
      metric("Forecast demand", number.format(service.total_units), label, "↗", true),
      metric("Recommended prep", number.format(totalPrep), `Units across ${items.length} items`, "▤"),
      metric("Demand level", service.demand_class, `Compared with past ${selectedService.toLowerCase()}s`, "◒"),
      metric("Shortage alerts", shortages.length, "At typical prep levels", "!")
    ].join("");
    byId("scenario-date").textContent = readableDate.format(toDate(selectedDate));
    for (const input of document.querySelectorAll("[data-flag]")) {
      input.checked = Boolean(flagsByDate.get(selectedDate)[input.dataset.flag]);
    }
    byId("prep-date").value = selectedDate;
    for (const button of document.querySelectorAll("[data-service]")) {
      button.setAttribute("aria-pressed", String(button.dataset.service === selectedService));
    }
    renderWeek(week);
    renderAlerts(service, shortages, label);
    renderPrep(items, label);
    renderWaste(items);
  }

  function renderBacktest() {
    const service = byId("backtest-service").value;
    const item = byId("backtest-item").value;
    const rows = replayRows.filter(row => (!service || row.time_period === service) && (!item || row.menu_item === item));
    const summary = backtest.summarize(rows, bundle.optimization_defaults);
    const formatError = error => error === null ? "—" : percent.format(error);
    byId("backtest-metrics").innerHTML = [
      metric("Item forecast WAPE", formatError(summary.wape), `Weekday baseline ${formatError(summary.baselineWape)}`, "↗", true),
      metric("Simulated shortage", number.format(summary.simulatedShortage), `Recorded ${number.format(summary.historicalShortage)} units`, "!"),
      metric("Simulated waste", number.format(summary.simulatedWaste), `Recorded ${number.format(summary.historicalWaste)} units`, "▤"),
      metric("Weighted penalty", number.format(summary.simulatedLoss), `Recorded ${number.format(summary.historicalLoss)}`, "◒")
    ].join("");
    const difference = summary.simulatedLoss - summary.historicalLoss;
    const outcome = difference === 0 ? "Matched the recorded penalty"
      : `${number.format(Math.abs(difference))} ${difference > 0 ? "higher" : "lower"} weighted penalty than recorded prep`;
    byId("backtest-result").textContent = `${outcome}. ${number.format(rows.length)} item records. WAPE is total absolute forecast error divided by actual demand.`;
    byId("backtest-rows").innerHTML = rows.map(row => `<tr>
      <td>${safe(row.date)}</td><td>${safe(row.time_period)}</td><td class="item-name">${safe(row.menu_item)}</td>
      <td>${number.format(row.actual_demand)}</td><td>${number.format(row.prediction)}</td>
      <td>${number.format(row.historical_prepared)}</td><td class="prep-quantity">${number.format(row.recommended)}</td>
      <td>${number.format(row.historical_shortage)} / ${number.format(row.historical_unsold)}</td>
      <td>${number.format(row.simulated_shortage)} / ${number.format(row.simulated_waste)}</td>
      </tr>`).join("");
  }

  function showPage() {
    const hash = window.location.hash.slice(1);
    if (hash === "main") return;
    const view = Object.hasOwn(pages, hash) ? hash : "forecast";
    const planning = view !== "backtest";
    byId("page-title").textContent = pages[view].title;
    byId("page-description").textContent = pages[view].description;
    byId("planning").hidden = !planning;
    byId("service-switch").hidden = !planning;
    byId("forecast-panels").hidden = view !== "forecast";
    byId("waste-panel").hidden = view !== "kitchen";
    byId("backtest-view").hidden = planning;
    byId("forecast-window").textContent = planning
      ? `Forecast · ${rangeDate.formatRange(toDate(dates[0]), toDate(dates.at(-1)))}`
      : `Holdout · ${rangeDate.formatRange(toDate(bundle.split.test_start), toDate(bundle.split.test_end))}`;
    for (const link of document.querySelectorAll("[data-view]")) {
      if (link.dataset.view === view) link.setAttribute("aria-current", "page");
      else link.removeAttribute("aria-current");
    }
    if (planning) renderPlanning();
    else renderBacktest();
  }

  function bindEvents() {
    window.addEventListener("hashchange", showPage);
    byId("service-switch").addEventListener("click", event => {
      const button = event.target.closest("[data-service]");
      if (!button) return;
      selectedService = button.dataset.service;
      renderPlanning();
    });
    byId("weekly-chart").addEventListener("click", event => {
      const button = event.target.closest("[data-date]");
      if (!button) return;
      selectedDate = button.dataset.date;
      renderPlanning();
    });
    byId("scenario-control").addEventListener("change", event => {
      flagsByDate.get(selectedDate)[event.target.dataset.flag] = Number(event.target.checked);
      renderPlanning();
    });
    byId("prep-date").addEventListener("change", event => {
      selectedDate = event.target.value;
      renderPlanning();
    });
    byId("backtest-service").addEventListener("change", renderBacktest);
    byId("backtest-item").addEventListener("change", renderBacktest);
  }

  async function start() {
    try {
      const [model, replay] = await Promise.all([readJson("./data/model.json"), readJson("./data/backtest.json")]);
      bundle = model;
      replayRows = backtest.replay(bundle, replay.records);
      dates = upcomingDates(new Date(), bundle.horizon_days);
      selectedDate = dates[0];
      for (const date of dates) {
        flagsByDate.set(date, { ...flagDefaults });
        byId("prep-date").add(new Option(readableDate.format(toDate(date)), date));
      }
      for (const item of Object.keys(bundle.models)) byId("backtest-item").add(new Option(item, item));
      bindEvents();
      if (window.location.hash === "#main") window.history.replaceState(null, "", "#forecast");
      showPage();
      byId("load-status").hidden = true;
      byId("dashboard").hidden = false;
    } catch (error) {
      console.error(error);
      byId("load-status").textContent = "The dashboard could not load its local data. Open it through the project's static server and refresh.";
    }
  }

  start();
})();
