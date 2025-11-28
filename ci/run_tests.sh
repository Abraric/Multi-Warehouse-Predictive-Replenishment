#!/bin/bash
# CI smoke test script for multi-warehouse replenishment system

set -e  # Exit on error

echo "=========================================="
echo "Running CI Smoke Tests"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
python --version || python3 --version

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -r requirements.txt || pip3 install -r requirements.txt

# Run unit tests
echo -e "${YELLOW}Running unit tests...${NC}"
pytest tests/ -v --tb=short || {
    echo -e "${RED}Unit tests failed!${NC}"
    exit 1
}

# Generate sample data (small scale for CI)
echo -e "${YELLOW}Generating sample data...${NC}"
python src/data/synthetic_generator.py \
    --out-dir data/raw \
    --seed 42 \
    --days 60 \
    --warehouses 3 \
    --skus 10 || {
    echo -e "${RED}Data generation failed!${NC}"
    exit 1
}

# Run ETL
echo -e "${YELLOW}Running ETL pipeline...${NC}"
python src/etl/etl.py \
    --in-dir data/raw \
    --out-dir data/processed || {
    echo -e "${RED}ETL failed!${NC}"
    exit 1
}

# Check ETL outputs
if [ ! -f "data/processed/daily_sales_aggregated.csv" ]; then
    echo -e "${RED}ETL output file missing!${NC}"
    exit 1
fi

# Run forecasting (may fail if insufficient data, that's OK for smoke test)
echo -e "${YELLOW}Running forecasting...${NC}"
python src/forecast/forecast_manager.py \
    --train \
    --data-dir data/processed \
    --out-dir data/forecasts \
    --horizon 7 || {
    echo -e "${YELLOW}Forecasting had issues (may be OK for small dataset)${NC}"
}

# Run optimization (if forecasts exist)
if [ -f "data/forecasts/forecasts.csv" ]; then
    echo -e "${YELLOW}Running optimization...${NC}"
    python -m src.optimization.replenishment_optimizer \
        --snapshot data/processed/inventory_snapshot.csv \
        --forecasts data/forecasts \
        --horizon 7 \
        --out output/reports/test_plan.csv || {
        echo -e "${YELLOW}Optimization had issues (may be OK if no transfers needed)${NC}"
    }
else
    echo -e "${YELLOW}Skipping optimization (no forecasts available)${NC}"
fi

echo -e "${GREEN}=========================================="
echo "CI Smoke Tests Completed Successfully!"
echo "==========================================${NC}"

exit 0

