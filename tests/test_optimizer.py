"""Tests for optimization engine."""

import pytest
import pandas as pd
from pathlib import Path
from datetime import datetime

from src.optimization.replenishment_optimizer import ReplenishmentOptimizer
from src.data.synthetic_generator import SyntheticDataGenerator


@pytest.fixture
def optimizer_data(tmp_path):
    """Create data for optimizer testing."""
    gen = SyntheticDataGenerator(seed=42)
    start_date = datetime.now() - timedelta(days=30)
    data = gen.generate_all(3, 5, 30, start_date)
    
    # Create simple forecasts
    forecasts = []
    for _, wh in data['warehouses'].iterrows():
        for _, sku in data['skus'].iterrows():
            for i in range(14):
                forecasts.append({
                    'date': (datetime.now() + timedelta(days=i)).date(),
                    'warehouse_id': wh['warehouse_id'],
                    'sku_id': sku['sku_id'],
                    'forecast': 10.0,
                    'lower': 8.0,
                    'upper': 12.0,
                })
    
    forecasts_df = pd.DataFrame(forecasts)
    
    # Save to temp
    snapshot_path = tmp_path / "snapshot.csv"
    data['inventory_snapshot'].to_csv(snapshot_path, index=False)
    
    forecast_path = tmp_path / "forecasts.csv"
    forecasts_df.to_csv(forecast_path, index=False)
    
    return {
        'snapshot': snapshot_path,
        'forecasts': forecast_path,
        'warehouses': data['warehouses'],
        'skus': data['skus'],
        'fleet': data['fleet'],
    }


def test_optimizer_initializes(optimizer_data):
    """Test optimizer initializes correctly."""
    optimizer = ReplenishmentOptimizer(
        optimizer_data['snapshot'],
        optimizer_data['forecasts'],
        optimizer_data['warehouses'],
        optimizer_data['skus'],
        optimizer_data['fleet'],
    )
    
    assert optimizer.plan_id is not None


def test_optimizer_generates_plan(optimizer_data):
    """Test optimizer generates a plan."""
    optimizer = ReplenishmentOptimizer(
        optimizer_data['snapshot'],
        optimizer_data['forecasts'],
        optimizer_data['warehouses'],
        optimizer_data['skus'],
        optimizer_data['fleet'],
    )
    
    plan = optimizer.optimize(horizon_days=7)
    
    assert isinstance(plan, pd.DataFrame)
    # Plan may be empty if no transfers needed, that's OK
    if len(plan) > 0:
        assert 'from_wh' in plan.columns
        assert 'to_wh' in plan.columns
        assert 'qty' in plan.columns


def test_optimizer_respects_perishability(optimizer_data):
    """Test optimizer prevents transfers violating TTL."""
    # Create SKU with very short TTL
    skus = optimizer_data['skus'].copy()
    skus.loc[skus.index[0], 'perishability_ttl_days'] = 1
    
    # Create warehouse with long lead time
    warehouses = optimizer_data['warehouses'].copy()
    warehouses.loc[warehouses.index[0], 'lead_time_days'] = 5
    
    optimizer = ReplenishmentOptimizer(
        optimizer_data['snapshot'],
        optimizer_data['forecasts'],
        warehouses,
        skus,
        optimizer_data['fleet'],
    )
    
    plan = optimizer.optimize(horizon_days=7)
    
    # Check no transfers violate TTL (if any transfers exist)
    if len(plan) > 0:
        for _, row in plan.iterrows():
            sku = skus[skus['sku_id'] == row['sku_id']].iloc[0]
            wh = warehouses[warehouses['warehouse_id'] == row['to_wh']].iloc[0]
            
            if pd.notna(sku.get('perishability_ttl_days')):
                ttl = sku['perishability_ttl_days']
                lead_time = wh['lead_time_days']
                assert lead_time <= ttl, "Transfer violates TTL constraint"


def test_greedy_heuristic_fallback(optimizer_data):
    """Test greedy heuristic works as fallback."""
    optimizer = ReplenishmentOptimizer(
        optimizer_data['snapshot'],
        optimizer_data['forecasts'],
        optimizer_data['warehouses'],
        optimizer_data['skus'],
        optimizer_data['fleet'],
    )
    
    plan = optimizer.greedy_heuristic(7, datetime.now())
    
    assert isinstance(plan, pd.DataFrame)

