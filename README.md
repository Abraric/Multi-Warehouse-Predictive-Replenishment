# Multi-Warehouse Predictive Replenishment with Transport Constraints

A production-ready system for multi-warehouse inventory management that combines demand forecasting, cost-optimal replenishment planning, and simulation capabilities.

## Features

- **Demand Forecasting**: Prophet and LightGBM models for SKU×warehouse level predictions
- **Optimization Engine**: OR-Tools/PuLP-based replenishment planner respecting transport constraints
- **Transfer Simulation**: Stochastic simulation with lead-time and demand variability
- **API & Dashboard**: FastAPI backend and Streamlit dashboard for operations teams
- **Synthetic Data**: Deterministic data generator for testing and demos

## Quick Start

```bash
# Install dependencies
make install

# Generate sample data
make data

# Run ETL pipeline
make etl

# Train forecasts
make forecast

# Generate replenishment plan
make optimize

# Run API server
make run

# Run tests
make test
```

## Project Structure

```
├── src/              # Core application code
├── notebooks/        # Demo notebooks
├── tests/            # Unit tests
├── data/             # Data directories (raw, processed)
├── ci/               # CI scripts
├── architecture/     # Architecture documentation
└── output/           # User-facing artifacts
```

## Requirements

- Python 3.9+
- Docker & Docker Compose (optional, for Postgres/Redis)

## Documentation

See `/output/README.md` for comprehensive user documentation and quickstart guide.

