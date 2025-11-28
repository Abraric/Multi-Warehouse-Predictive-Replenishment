# Quickstart Guide

## One-Command Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete pipeline
make data && make etl && make forecast && make optimize && make simulate
```

## Step-by-Step

### 1. Generate Data
```bash
python src/data/synthetic_generator.py \
    --out-dir data/raw \
    --seed 42 \
    --days 180 \
    --warehouses 5 \
    --skus 50
```

### 2. Run ETL
```bash
python src/etl/etl.py \
    --in-dir data/raw \
    --out-dir data/processed
```

### 3. Train Forecasts
```bash
python src/forecast/forecast_manager.py \
    --train \
    --data-dir data/processed \
    --out-dir data/forecasts \
    --horizon 14
```

### 4. Optimize
```bash
python -m src.optimization.replenishment_optimizer \
    --snapshot data/processed/inventory_snapshot.csv \
    --forecasts data/forecasts \
    --horizon 14 \
    --out output/reports/replenishment_plan_sample.csv
```

### 5. Simulate
```bash
python src/simulator/transfer_simulator.py \
    --plan output/reports/replenishment_plan_sample.csv \
    --out output/reports/simulation_report_sample.csv
```

## API Usage

### Start API Server
```bash
uvicorn src.api.app:app --reload --port 8000
```

### Example API Calls

```bash
# Get forecast
curl "http://localhost:8000/forecast?warehouse_id=WH-001&sku_id=SKU-0001&horizon=14"

# Run optimization
curl -X POST "http://localhost:8000/optimize" \
    -H "Content-Type: application/json" \
    -d '{"horizon_days": 14}'

# Get plan
curl "http://localhost:8000/plan/PLAN-20240101-120000-abc123"

# Get metrics
curl "http://localhost:8000/metrics?plan_id=PLAN-20240101-120000-abc123"

# Simulate plan
curl -X POST "http://localhost:8000/simulate/PLAN-20240101-120000-abc123"
```

## Dashboard Usage

### Start Dashboard
```bash
streamlit run src/dashboard/app.py --server.port 8501
```

Open `http://localhost:8501` in your browser.

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run CI smoke test
bash ci/run_tests.sh
```

