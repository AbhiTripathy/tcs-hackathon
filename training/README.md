# Demand models

Create a Python 3.12 environment, install `training/requirements.txt`, and run
`python training/train.py` from the repository root. The script resolves default
paths relative to itself. Optional `--data-dir` and `--output-dir` accept paths.

The generated files are `web/data/model.json` and `web/data/backtest.json`.
Training validates the input counts, unique keys, complete service grid, calendar
coverage, and reconciliation between orders and preparation outcomes.

## Training and evaluation

Each item has a scikit-learn Ridge regression with alpha 1 and the SVD solver.
The shared feature order is Dinner indicator, Tuesday through Sunday indicators
(Monday is the reference), holiday, event, and promotion indicators. Coefficients
use raw binary features, with an intercept. Clip the final prediction at zero.
Keep forecasts unrounded until the independent prep optimizer chooses integers.

All item and service records for a date stay in the same partition.

- Training runs January 1 through April 30, 2025.
- Calibration runs May 1 through May 30, 2025.
- The holdout runs May 31 through June 29, 2025.

Parameters, reference prep quantities, historical waste summaries, and demand
thresholds use training data only. Calibration residuals use the frozen model.
Holdout outcomes enter scoring only. Models remain frozen after evaluation so
the exported model reproduces the reported results. There is no tuning on the
holdout. The comparison baseline is each item's training weekday/service mean.

WAPE is summed absolute error divided by summed actual demand. MAE and bias are
in item units. Service metrics sum item forecasts before computing errors.

## Browser contract

Load `web/js/forecast.js` as a classic script. It exposes `RushIQForecast`.
Load the JSON using the frontend's static asset loader, then call
`RushIQForecast.forecastService(bundle, date, service, flags)`.
Dates use `YYYY-MM-DD`, and services use `Lunch` or `Dinner`. Supply all three
scenario flags explicitly as numeric 0 or 1. Use `buildFeatures` and `predictItem`
for individual forecasts or replay. Feature weekdays use UTC date parsing and
Monday=0. Each item prediction includes signed contributions and its intercept
for an additive explanation. Contributions describe associations in this model.

Generate seven dates starting tomorrow and call `forecastService` for each
service. The frontend owns the restaurant's definition of today's date and the
scenario flags for each date. Calendar features permit direct forecasts for the
entire week without consuming future actual sales or recursive predictions.

Demand classes use service totals with training 25th and 75th percentile
thresholds. Equality at either threshold is Normal. Spike alerts use the 90th
percentile. All thresholds and classification rules are in the bundle.

## Optimizer inputs

For an item/service, `models[item].services[service]` contains 30 calibration
residuals, typical prepared units (training median), historical prepared and
unsold totals, waste rate, and weekday mean demand. Historical summaries cover
the training partition. The defaults specify shortage penalty 2 and waste penalty 1.

The separate optimizer can form scenarios as `max(0, prediction + residual)`
and minimize mean `2 * max(demand - quantity, 0) + max(quantity - demand, 0)`
over nonnegative integer quantities. Risk is the empirical fraction of scenarios
with demand above or below the chosen quantity. These are empirical estimates
from 30 observations per item/service, with no guaranteed coverage for future
distribution changes. An optimizer should explain the penalty and any rounding.

Replay records contain features, actual demand, historical prepared/unsold/
shortage units, baseline predictions, and expected Python predictions. Build
recommendations from features and the model before reading the outcome fields.
Score simulated shortages and leftovers against actual demand for each record.
Actual recommended-policy outcomes remain a simulation.

The data is synthetic and ends June 29, 2025. Forecasts for current dates assume
the historical weekly and calendar relationships persist. There are few holiday
and event examples. Item units proxy for demand and unsold units proxy for waste.
Actual customer counts, independent lost sales, and discarded food are unavailable.
