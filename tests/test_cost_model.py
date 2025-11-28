"""Tests for cost model utilities."""

import pytest
import pandas as pd
import numpy as np

from src.utils.cost_model import CostModel
from src.data.synthetic_generator import SyntheticDataGenerator


@pytest.fixture
def cost_model_data():
    """Create data for cost model testing."""
    gen = SyntheticDataGenerator(seed=42)
    warehouses = gen.generate_warehouses(3)
    skus = gen.generate_skus(5)
    fleet = gen.generate_fleet()
    
    return warehouses, skus, fleet


def test_cost_model_initializes(cost_model_data):
    """Test cost model initializes."""
    warehouses, skus, fleet = cost_model_data
    cost_model = CostModel(warehouses, skus, fleet)
    
    assert cost_model.warehouses is not None
    assert cost_model.skus is not None


def test_transport_cost_calculation(cost_model_data):
    """Test transport cost calculation."""
    warehouses, skus, fleet = cost_model_data
    cost_model = CostModel(warehouses, skus, fleet)
    
    wh1 = warehouses.iloc[0]['warehouse_id']
    wh2 = warehouses.iloc[1]['warehouse_id']
    sku_id = skus.iloc[0]['sku_id']
    
    cost = cost_model.transport_cost(wh1, wh2, 100, sku_id, 'Medium')
    
    assert cost > 0
    assert isinstance(cost, (int, float))


def test_holding_cost_calculation(cost_model_data):
    """Test holding cost calculation."""
    warehouses, skus, fleet = cost_model_data
    cost_model = CostModel(warehouses, skus, fleet)
    
    wh_id = warehouses.iloc[0]['warehouse_id']
    sku_id = skus.iloc[0]['sku_id']
    
    cost = cost_model.holding_cost(wh_id, sku_id, 100, 7)
    
    assert cost >= 0
    assert isinstance(cost, (int, float))


def test_stockout_cost_calculation(cost_model_data):
    """Test stockout cost calculation."""
    warehouses, skus, fleet = cost_model_data
    cost_model = CostModel(warehouses, skus, fleet)
    
    wh_id = warehouses.iloc[0]['warehouse_id']
    sku_id = skus.iloc[0]['sku_id']
    
    cost = cost_model.stockout_cost(wh_id, sku_id, 50)
    
    assert cost > 0
    assert isinstance(cost, (int, float))


def test_perishability_waste_cost(cost_model_data):
    """Test perishability waste cost calculation."""
    warehouses, skus, fleet = cost_model_data
    cost_model = CostModel(warehouses, skus, fleet)
    
    # Find a perishable SKU or create one
    skus_test = skus.copy()
    skus_test.loc[skus_test.index[0], 'perishability_ttl_days'] = 5
    skus_test.loc[skus_test.index[0], 'unit_cost'] = 10.0
    
    cost_model_test = CostModel(warehouses, skus_test, fleet)
    sku_id = skus_test.iloc[0]['sku_id']
    
    # Test with TTL violation
    waste_cost = cost_model_test.perishability_waste_cost(sku_id, 100, 10)  # 10 days > 5 day TTL
    assert waste_cost > 0
    
    # Test without violation
    no_waste_cost = cost_model_test.perishability_waste_cost(sku_id, 100, 2)  # 2 days < 5 day TTL
    assert no_waste_cost == 0

