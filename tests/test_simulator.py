"""Tests for transfer simulator."""

import pytest
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

from src.simulator.transfer_simulator import TransferSimulator
from src.data.synthetic_generator import SyntheticDataGenerator


@pytest.fixture
def simulator_data(tmp_path):
    """Create data for simulator testing."""
    gen = SyntheticDataGenerator(seed=42)
    start_date = datetime.now() - timedelta(days=30)
    data = gen.generate_all(3, 5, 30, start_date)
    
    # Create simple plan
    plan_rows = []
    for i in range(3):
        plan_rows.append({
            'plan_id': 'TEST-PLAN',
            'run_date': datetime.now().date(),
            'from_wh': data['warehouses'].iloc[0]['warehouse_id'],
            'to_wh': data['warehouses'].iloc[1]['warehouse_id'],
            'sku_id': data['skus'].iloc[i]['sku_id'],
            'qty': 100,
            'ship_date': datetime.now().date(),
            'expected_arrival': (datetime.now() + timedelta(days=2)).date(),
            'truck_type': 'Medium',
            'driver_id': None,
            'estimated_cost': 100.0,
            'reason_code': 'TEST',
        })
    
    plan = pd.DataFrame(plan_rows)
    plan_path = tmp_path / "plan.csv"
    plan.to_csv(plan_path, index=False)
    
    # Create forecasts
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
    forecast_path = tmp_path / "forecasts.csv"
    forecasts_df.to_csv(forecast_path, index=False)
    
    return {
        'plan': plan_path,
        'warehouses': data['warehouses'],
        'skus': data['skus'],
        'fleet': data['fleet'],
        'inventory': data['inventory_snapshot'],
        'forecasts': forecasts_df,
    }


def test_simulator_initializes(simulator_data):
    """Test simulator initializes correctly."""
    simulator = TransferSimulator(
        simulator_data['plan'],
        simulator_data['warehouses'],
        simulator_data['skus'],
        simulator_data['fleet'],
        seed=42
    )
    
    assert simulator.plan is not None
    assert len(simulator.plan) > 0


def test_simulator_generates_report(simulator_data):
    """Test simulator generates report."""
    simulator = TransferSimulator(
        simulator_data['plan'],
        simulator_data['warehouses'],
        simulator_data['skus'],
        simulator_data['fleet'],
        seed=42
    )
    
    report = simulator.generate_report(
        simulator_data['inventory'],
        simulator_data['forecasts']
    )
    
    assert 'summary' in report
    assert 'transfers' in report
    assert 'stockouts' in report
    
    assert report['summary']['total_transfers'] > 0


def test_simulator_respects_seed(simulator_data):
    """Test simulator produces deterministic results with same seed."""
    sim1 = TransferSimulator(
        simulator_data['plan'],
        simulator_data['warehouses'],
        simulator_data['skus'],
        simulator_data['fleet'],
        seed=42
    )
    
    sim2 = TransferSimulator(
        simulator_data['plan'],
        simulator_data['warehouses'],
        simulator_data['skus'],
        simulator_data['fleet'],
        seed=42
    )
    
    report1 = sim1.generate_report(simulator_data['inventory'], simulator_data['forecasts'])
    report2 = sim2.generate_report(simulator_data['inventory'], simulator_data['forecasts'])
    
    # Summary should be identical with same seed
    assert report1['summary']['total_transport_cost'] == report2['summary']['total_transport_cost']

