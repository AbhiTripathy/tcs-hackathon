# RushIQ synthetic hackathon dataset

All records in this package are synthetic and generated for development/testing. They are not real restaurant records.

## Run the dashboard

From the repository root, serve the static `web/` directory with Python:

```powershell
python -m http.server 8765 --bind 127.0.0.1 --directory web
```

If you use the project's Windows virtual environment, replace `python` with
`.\.venv\Scripts\python.exe`. Open [the local dashboard](http://127.0.0.1:8765/).
The browser loads the exported model and holdout data from `web/data/`.
See [the training guide](training/README.md) to regenerate the exports.

## Files
- `orders.csv`: order quantities by date, time period, menu item and order channel.
- `menu_preparation.csv`: quantities prepared/sold, unsold and shortage quantities, and preparation time by date, period and item.
- `calendar.csv`: weekday/weekend, holiday/event and promotion indicators.

## Coverage
- Dates: 2025-01-01 through 2025-06-29 (180 days)
- Time periods: Lunch and Dinner
- Menu items: 8
- Order channels: Dine-in, Takeaway, Delivery

## Important relationships
- Summing `quantity_ordered` in orders.csv by date + time_period + menu_item gives `quantity_sold` in menu_preparation.csv.
- `unsold_quantity = max(quantity_prepared - quantity_sold, 0)`.
- `shortage_quantity = max(quantity_sold - quantity_prepared, 0)`.
- Calendar events and promotions are synthetic scenario markers, not a verified public-holiday calendar.

## Suggested use
1. Aggregate orders by date, time_period and menu_item.
2. Join calendar features by date.
3. Build forecasts using only information available before the prediction date.
4. Evaluate chronologically (train on earlier dates, test on later dates); avoid random train/test splitting for time-series forecasting.
5. Compare predictions with actual `quantity_sold` and report MAE or WAPE.

## Columns
### orders.csv
`date`, `time_period`, `menu_item`, `item_category`, `quantity_ordered`, `order_channel`

### menu_preparation.csv
`date`, `time_period`, `menu_item`, `item_category`, `quantity_prepared`, `quantity_sold`, `unsold_quantity`, `shortage_quantity`, `preparation_time_minutes`, `synthetic_waste_flag`

### calendar.csv
`date`, `day_of_week`, `is_weekend`, `is_holiday`, `holiday_name`, `is_festival_or_event`, `event_name`, `is_promotion`, `promotion_type`, `notes`
