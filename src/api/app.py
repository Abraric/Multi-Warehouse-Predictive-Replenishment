"""
FastAPI backend for multi-warehouse replenishment system.

Exposes endpoints for forecasts, optimization, plan retrieval, metrics, and simulation.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import logging

from src.forecast.forecast_manager import ForecastManager
from src.optimization.replenishment_optimizer import ReplenishmentOptimizer
from src.simulator.transfer_simulator import TransferSimulator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Multi-Warehouse Replenishment API", version="1.0.0")

# Global state (in production, use proper state management)
DATA_DIR = Path("data/processed")
FORECASTS_DIR = Path("data/forecasts")
PLANS_DIR = Path("output/reports")


class ForecastRequest(BaseModel):
    warehouse_id: str
    sku_id: str
    horizon: int = 14
    model_type: Optional[str] = "prophet"


class OptimizeRequest(BaseModel):
    horizon_days: int = 14
    start_date: Optional[str] = None


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Multi-Warehouse Replenishment API",
        "version": "1.0.0",
        "endpoints": [
            "/forecast",
            "/optimize",
            "/plan/{plan_id}",
            "/metrics",
            "/simulate/{plan_id}",
        ]
    }


@app.get("/forecast")
async def get_forecast(
    warehouse_id: str = Query(..., description="Warehouse ID"),
    sku_id: str = Query(..., description="SKU ID"),
    horizon: int = Query(14, description="Forecast horizon in days"),
    model_type: str = Query("prophet", description="Model type: prophet or lightgbm")
):
    """Get demand forecast for a specific SKU×warehouse."""
    try:
        manager = ForecastManager(DATA_DIR, FORECASTS_DIR)
        
        # Load models if available
        models_file = FORECASTS_DIR / "models"
        if not models_file.exists():
            raise HTTPException(
                status_code=404,
                detail="Models not found. Please train models first."
            )
        
        start_date = datetime.now()
        forecast = manager.predict(warehouse_id, sku_id, start_date, horizon, model_type)
        
        if forecast.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No forecast available for {warehouse_id} × {sku_id}"
            )
        
        return {
            "warehouse_id": warehouse_id,
            "sku_id": sku_id,
            "horizon": horizon,
            "forecast": forecast.to_dict(orient="records")
        }
    except Exception as e:
        logger.error(f"Error getting forecast: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimize")
async def optimize_replenishment(request: OptimizeRequest):
    """Run optimization and generate replenishment plan."""
    try:
        # Load required data
        warehouses = pd.read_csv(DATA_DIR / "warehouses.csv")
        skus = pd.read_csv(DATA_DIR / "skus.csv")
        fleet_path = Path("data/raw/fleet.csv")
        fleet = pd.read_csv(fleet_path) if fleet_path.exists() else pd.DataFrame()
        
        snapshot_path = DATA_DIR / "inventory_snapshot.csv"
        forecasts_path = FORECASTS_DIR / "forecasts.csv"
        
        if not snapshot_path.exists():
            raise HTTPException(status_code=404, detail="Inventory snapshot not found")
        if not forecasts_path.exists():
            raise HTTPException(status_code=404, detail="Forecasts not found")
        
        # Parse start date
        start_date = datetime.now()
        if request.start_date:
            start_date = datetime.fromisoformat(request.start_date)
        
        # Create optimizer
        optimizer = ReplenishmentOptimizer(
            snapshot_path,
            forecasts_path,
            warehouses,
            skus,
            fleet
        )
        
        # Optimize
        plan = optimizer.optimize(request.horizon_days, start_date)
        
        # Save plan
        plan_id = optimizer.plan_id
        plan_path = PLANS_DIR / f"{plan_id}.csv"
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan.to_csv(plan_path, index=False)
        
        return {
            "plan_id": plan_id,
            "transfers": len(plan),
            "total_cost": plan['estimated_cost'].sum() if len(plan) > 0 else 0,
            "plan": plan.to_dict(orient="records")
        }
    except Exception as e:
        logger.error(f"Error optimizing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/plan/{plan_id}")
async def get_plan(plan_id: str):
    """Get replenishment plan details."""
    try:
        plan_path = PLANS_DIR / f"{plan_id}.csv"
        
        if not plan_path.exists():
            raise HTTPException(status_code=404, detail=f"Plan {plan_id} not found")
        
        plan = pd.read_csv(plan_path)
        
        return {
            "plan_id": plan_id,
            "transfers": len(plan),
            "total_cost": plan['estimated_cost'].sum() if len(plan) > 0 else 0,
            "plan": plan.to_dict(orient="records")
        }
    except Exception as e:
        logger.error(f"Error getting plan: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics(plan_id: Optional[str] = None):
    """Get key KPIs and metrics."""
    try:
        if plan_id:
            plan_path = PLANS_DIR / f"{plan_id}.csv"
            if not plan_path.exists():
                raise HTTPException(status_code=404, detail=f"Plan {plan_id} not found")
            plan = pd.read_csv(plan_path)
        else:
            # Get latest plan
            plan_files = sorted(PLANS_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not plan_files:
                return {
                    "message": "No plans found",
                    "metrics": {}
                }
            plan = pd.read_csv(plan_files[0])
            plan_id = plan_files[0].stem
        
        # Calculate metrics
        total_cost = plan['estimated_cost'].sum() if len(plan) > 0 else 0
        total_transfers = len(plan)
        
        # Estimate stockouts (simplified)
        # In production, this would use actual simulation results
        expected_stockouts = 0  # Placeholder
        
        # Estimate waste (simplified)
        skus = pd.read_csv(DATA_DIR / "skus.csv")
        plan_with_skus = plan.merge(skus[['sku_id', 'perishability_ttl_days']], on='sku_id', how='left')
        perishable_transfers = plan_with_skus[plan_with_skus['perishability_ttl_days'].notna()]
        expected_waste = len(perishable_transfers) * 0.1  # Placeholder: 10% waste rate
        
        return {
            "plan_id": plan_id,
            "metrics": {
                "total_cost": float(total_cost),
                "total_transfers": total_transfers,
                "expected_stockouts": expected_stockouts,
                "expected_waste": expected_waste,
                "service_level": 1.0 - (expected_stockouts / max(total_transfers, 1)),
            }
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/simulate/{plan_id}")
async def simulate_plan(plan_id: str):
    """Run simulation for a replenishment plan."""
    try:
        plan_path = PLANS_DIR / f"{plan_id}.csv"
        if not plan_path.exists():
            raise HTTPException(status_code=404, detail=f"Plan {plan_id} not found")
        
        # Load data
        warehouses = pd.read_csv(DATA_DIR / "warehouses.csv")
        skus = pd.read_csv(DATA_DIR / "skus.csv")
        fleet_path = Path("data/raw/fleet.csv")
        fleet = pd.read_csv(fleet_path) if fleet_path.exists() else pd.DataFrame()
        inventory = pd.read_csv(DATA_DIR / "inventory_snapshot.csv")
        forecasts = pd.read_csv(FORECASTS_DIR / "forecasts.csv")
        
        # Create simulator
        simulator = TransferSimulator(plan_path, warehouses, skus, fleet)
        
        # Run simulation
        report = simulator.generate_report(inventory, forecasts)
        
        # Save report
        report_path = PLANS_DIR / f"{plan_id}_simulation.csv"
        report['transfers'].to_csv(report_path, index=False)
        
        return {
            "plan_id": plan_id,
            "simulation": report['summary'],
            "transfers": report['transfers'].to_dict(orient="records"),
            "stockouts": report['stockouts'].to_dict(orient="records") if len(report['stockouts']) > 0 else [],
        }
    except Exception as e:
        logger.error(f"Error simulating plan: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

