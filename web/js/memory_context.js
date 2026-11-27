/* RushIQ Memory Context — browser-side companion to memory_store.py
 *
 * Provides a lightweight in-browser memory store that mirrors the Python
 * short-term and episodic layers using localStorage. The semantic layer
 * is derived live from the already-loaded model bundle (never duplicated).
 *
 * This file is a classic script; load it AFTER forecast.js.
 * It does NOT modify RushIQForecast or any existing behaviour.
 *
 * Usage:
 *   const ctx = RushIQMemory.startSession("Forecast for 2025-07-01");
 *   const result = RushIQForecast.forecastService(bundle, date, service, flags);
 *   RushIQMemory.savePrediction(date, service, result, flags);
 *   const context = RushIQMemory.contextFor(bundle, date, service);
 *   // ... display context.episodic, context.semantic etc.
 */
(function (global) {
  "use strict";

  var STORAGE_KEY_EPISODIC  = "rushiq_episodic_v1";
  var STORAGE_KEY_LONG_TERM = "rushiq_long_term_v1";
  var MAX_EPISODIC = 90;      // keep at most 90 records
  var MAX_EPISODIC_DAYS = 90; // prune records older than 90 days

  // -------------------------------------------------------------------------
  // Persistence helpers (localStorage; degrade gracefully if unavailable)
  // -------------------------------------------------------------------------
  function safeRead(key, fallback) {
    try {
      var raw = localStorage.getItem(key);
      return raw ? JSON.parse(raw) : fallback;
    } catch (e) {
      return fallback;
    }
  }

  function safeWrite(key, data) {
    try {
      localStorage.setItem(key, JSON.stringify(data));
      return true;
    } catch (e) {
      return false;
    }
  }

  function nowIso() {
    return new Date().toISOString();
  }

  // -------------------------------------------------------------------------
  // Short-term memory (session-scoped, in-memory only)
  // -------------------------------------------------------------------------
  var _shortTerm = {};

  var ShortTerm = {
    set: function (key, value) { _shortTerm[key] = value; },
    get: function (key, def)   { return Object.prototype.hasOwnProperty.call(_shortTerm, key) ? _shortTerm[key] : def; },
    clear: function ()         { _shortTerm = {}; },
    snapshot: function ()      { return Object.assign({}, _shortTerm); },
  };

  // -------------------------------------------------------------------------
  // Long-term memory (localStorage)
  // -------------------------------------------------------------------------
  var _defaultLongTerm = {
    schema_version: 1,
    preferences: { shortage_penalty: 2, waste_penalty: 1, default_forecast_horizon_days: 7 },
    operational_rules: [
      "Calendar flags must be supplied explicitly as 0 or 1.",
      "Forecasts are unrounded until the prep optimizer chooses integers.",
      "Data is synthetic; item units proxy demand, unsold units proxy waste.",
    ],
    updated_at: null,
  };

  var LongTerm = {
    _data: null,
    _load: function () {
      if (!this._data) {
        this._data = safeRead(STORAGE_KEY_LONG_TERM, Object.assign({}, _defaultLongTerm));
      }
      return this._data;
    },
    get: function (key) { return this._load()[key]; },
    updatePreference: function (key, value) {
      var d = this._load();
      d.preferences = d.preferences || {};
      d.preferences[key] = value;
      d.updated_at = nowIso();
      return safeWrite(STORAGE_KEY_LONG_TERM, d);
    },
    addRule: function (rule) {
      var d = this._load();
      d.operational_rules = d.operational_rules || [];
      if (d.operational_rules.indexOf(rule) === -1) {
        d.operational_rules.push(rule);
        d.updated_at = nowIso();
        return safeWrite(STORAGE_KEY_LONG_TERM, d);
      }
      return false;
    },
    snapshot: function () { return Object.assign({}, this._load()); },
  };

  // -------------------------------------------------------------------------
  // Episodic memory (localStorage, pruned on load)
  // -------------------------------------------------------------------------
  var EpisodicStore = {
    _records: null,
    _load: function () {
      if (!this._records) {
        var raw = safeRead(STORAGE_KEY_EPISODIC, { records: [] });
        this._records = raw.records || [];
        this._prune();
      }
      return this._records;
    },
    _prune: function () {
      var cutoff = Date.now() - MAX_EPISODIC_DAYS * 86400 * 1000;
      this._records = this._records.filter(function (r) {
        return new Date(r.predicted_at || 0).getTime() >= cutoff;
      });
      if (this._records.length > MAX_EPISODIC) {
        this._records = this._records.slice(-MAX_EPISODIC);
      }
    },
    _save: function () {
      return safeWrite(STORAGE_KEY_EPISODIC, { schema_version: 1, records: this._records });
    },
    savePrediction: function (date, service, forecastResult, flags) {
      var records = this._load();
      var existing = records.find(function (r) { return r.date === date && r.service === service; });
      if (existing) return false;
      var predictions = {};
      (forecastResult.items || []).forEach(function (it) {
        predictions[it.menu_item] = it.predicted_units;
      });
      records.push({
        date: date, service: service,
        predictions: predictions,
        total_predicted: forecastResult.total_units || 0,
        demand_class: forecastResult.demand_class,
        spike_alert: forecastResult.spike_alert,
        flags: flags,
        actual_demand: null, total_actual: null, mae: null, bias: null,
        predicted_at: nowIso(), outcome_recorded_at: null,
      });
      return this._save();
    },
    recordOutcome: function (date, service, actualDemand) {
      var records = this._load();
      var rec = records.find(function (r) { return r.date === date && r.service === service; });
      if (!rec) return false;
      var preds = rec.predictions || {};
      var items = Object.keys(actualDemand);
      var errors = items.map(function (k) { return Math.abs((preds[k] || 0) - actualDemand[k]); });
      var biases = items.map(function (k) { return (preds[k] || 0) - actualDemand[k]; });
      rec.actual_demand = actualDemand;
      rec.total_actual  = items.reduce(function (s, k) { return s + actualDemand[k]; }, 0);
      rec.mae   = errors.length ? errors.reduce(function (a, b) { return a + b; }, 0) / errors.length : null;
      rec.bias  = biases.length ? biases.reduce(function (a, b) { return a + b; }, 0) / biases.length : null;
      rec.outcome_recorded_at = nowIso();
      return this._save();
    },
    recent: function (n) {
      var r = this._load();
      return r.slice(-( n || 10)).reverse();
    },
    forDateService: function (date, service) {
      var r = this._load();
      for (var i = r.length - 1; i >= 0; i--) {
        if (r[i].date === date && r[i].service === service) return r[i];
      }
      return null;
    },
    averageMae: function (service) {
      var r = this._load();
      var candidates = r.filter(function (rec) {
        return rec.mae !== null && (service == null || rec.service === service);
      });
      if (!candidates.length) return null;
      return candidates.reduce(function (s, rec) { return s + rec.mae; }, 0) / candidates.length;
    },
  };

  // -------------------------------------------------------------------------
  // Semantic memory  (derived from model bundle — never duplicated)
  // -------------------------------------------------------------------------
  function buildSemantic(bundle) {
    if (!bundle) return {};
    return {
      menu_items: Object.keys(bundle.models || {}).sort(),
      services: ["Lunch", "Dinner"],
      feature_order: bundle.feature_order || [],
      feature_rules: bundle.feature_rules || {},
      service_thresholds: bundle.service_thresholds || {},
      optimization_defaults: bundle.optimization_defaults || { shortage_penalty: 2, waste_penalty: 1 },
      limitations: bundle.limitations || [],
      data_range: { start: bundle.data_start, end: bundle.data_end },
    };
  }

  // -------------------------------------------------------------------------
  // Context assembly (prioritised, size-capped)
  // -------------------------------------------------------------------------
  function contextFor(bundle, date, service, options) {
    var opts = options || {};
    var maxEpisodic = opts.maxEpisodic || 10;
    var ctx = {};

    // P1 — Semantic
    ctx.semantic = buildSemantic(bundle);

    // P2 — Short-term (current session)
    var st = ShortTerm.snapshot();
    if (Object.keys(st).length) ctx.current_session = st;

    // P3 — Episodic
    var recent = EpisodicStore.recent(maxEpisodic);
    var verified   = recent.filter(function (r) { return r.actual_demand !== null; });
    var unverified = recent.filter(function (r) { return r.actual_demand === null; });
    var episodicList = [];
    var seen = {};
    if (date && service) {
      var exact = EpisodicStore.forDateService(date, service);
      if (exact) { episodicList.push(exact); seen[date + "|" + service] = true; }
    }
    verified.concat(unverified).forEach(function (r) {
      var k = r.date + "|" + r.service;
      if (!seen[k]) { episodicList.push(r); seen[k] = true; }
    });
    ctx.episodic = {
      records: episodicList.slice(0, maxEpisodic),
      average_mae: EpisodicStore.averageMae(service),
      note: "Predictions and outcomes are stored separately. Outcomes are null until verified.",
    };

    // P4 — Long-term
    var lt = LongTerm.snapshot();
    ctx.long_term = {
      preferences: lt.preferences || {},
      operational_rules: lt.operational_rules || [],
    };

    return ctx;
  }

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------
  global.RushIQMemory = Object.freeze({
    // Session
    startSession: function (userRequest) {
      ShortTerm.clear();
      ShortTerm.set("user_request", userRequest || "");
      ShortTerm.set("session_started_at", nowIso());
      ShortTerm.set("state", "active");
    },
    endSession: function () {
      ShortTerm.set("state", "complete");
      ShortTerm.set("session_ended_at", nowIso());
    },

    // Short-term
    setShortTerm: function (key, value) { ShortTerm.set(key, value); },
    getShortTerm: function (key, def)   { return ShortTerm.get(key, def); },

    // Long-term
    updatePreference: function (key, value) { return LongTerm.updatePreference(key, value); },
    addRule: function (rule) { return LongTerm.addRule(rule); },

    // Episodic
    savePrediction: function (date, service, forecastResult, flags) {
      return EpisodicStore.savePrediction(date, service, forecastResult, flags);
    },
    recordOutcome: function (date, service, actualDemand) {
      return EpisodicStore.recordOutcome(date, service, actualDemand);
    },
    recentPredictions: function (n) { return EpisodicStore.recent(n); },

    // Context retrieval
    contextFor: function (bundle, date, service, options) {
      return contextFor(bundle, date, service, options);
    },

    // Semantic (live from bundle)
    semanticSummary: function (bundle) { return buildSemantic(bundle); },

    // Observability
    status: function (bundle) {
      return {
        short_term_keys: Object.keys(ShortTerm.snapshot()),
        long_term_updated_at: LongTerm.get("updated_at"),
        episodic_record_count: EpisodicStore._load().length,
        episodic_avg_mae_overall: EpisodicStore.averageMae(null),
        semantic_menu_items: bundle ? Object.keys(bundle.models || {}).sort() : [],
        storage: typeof localStorage !== "undefined" ? "localStorage" : "unavailable",
      };
    },
  });

})(globalThis);
