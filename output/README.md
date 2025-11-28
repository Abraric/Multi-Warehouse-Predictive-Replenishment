# Multi-Warehouse Predictive Replenishment — Demo Output

## What this demo contains

- `sample_data/` — sample raw CSVs (warehouses, SKUs, historical sales)

- `replenishment_plan_sample.csv` — example optimizer output

- `simulation_report_sample.csv` — example simulation run

- Visuals: forecast chart, plan gantt, cost breakdown



## Quick highlights

- Demonstrates demand forecasting per SKU×warehouse (7/14/28d horizons)

- Produces multi-warehouse cost-optimal replenishment plans respecting truck & driver constraints and perishability TTL

- Includes simulation to evaluate plan robustness under stochastic delays and demand shocks

- Provides API & Streamlit dashboard for operations teams to explore forecasts, plans, and simulations



## Quickstart (short)

1. Install Python dependencies: `pip install -r requirements.txt`

2. Generate sample data:  

   `python src/data/synthetic_generator.py --out-dir data/raw --seed 42 --days 180 --warehouses 5 --skus 50`

3. Run ETL:  

   `python src/etl/etl.py --in-dir data/raw --out-dir data/processed`

4. Train forecasts / produce forecasts:  

   `python src/forecast/forecast_manager.py --train --data-dir data/processed --out-dir data/forecasts --horizon 14`

5. Run optimizer:  

   `python -m src.optimization.replenishment_optimizer --snapshot data/processed/inventory_snapshot.csv --forecasts data/forecasts --horizon 14 --out output/reports/replenishment_plan_sample.csv`

6. Simulate the plan:  

   `python src/simulator/transfer_simulator.py --plan output/reports/replenishment_plan_sample.csv --out output/reports/simulation_report_sample.csv`

7. Run API + Dashboard:  

   `uvicorn src.api.app:app --reload --port 8000`  

   `streamlit run src.dashboard.app.py --server.port 8501`

8. Open Streamlit at `http://localhost:8501`



## What to inspect

- `replenishment_plan_sample.csv`: arrival dates, truck assignments, expected cost

- `simulation_report_sample.csv`: realized service level, cost variance, waste due to perishability

- Streamlit dashboard: interactive forecast explorer, plan Gantt, what-if scenario results



## Notes & production next steps

- For production: connect to real data lake, deploy solver to scalable worker, integrate with TMS/WMS, add RBAC & audit logs, and include monitoring of model drift.

