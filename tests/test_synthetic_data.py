"""Tests for synthetic data generator."""

import pytest
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

from src.data.synthetic_generator import SyntheticDataGenerator


def test_synthetic_generator_seed():
    """Test that generator produces deterministic results with same seed."""
    gen1 = SyntheticDataGenerator(seed=42)
    gen2 = SyntheticDataGenerator(seed=42)
    
    data1 = gen1.generate_all(5, 10, 30)
    data2 = gen2.generate_all(5, 10, 30)
    
    # Check warehouses are identical
    pd.testing.assert_frame_equal(data1['warehouses'], data2['warehouses'])
    assert len(data1['warehouses']) == 5


def test_warehouse_data_structure():
    """Test warehouse data has required columns."""
    gen = SyntheticDataGenerator(seed=42)
    warehouses = gen.generate_warehouses(3)
    
    required_cols = ['warehouse_id', 'name', 'capacity_volume_m3', 'lead_time_days']
    assert all(col in warehouses.columns for col in required_cols)
    assert len(warehouses) == 3


def test_sku_data_structure():
    """Test SKU data has required columns."""
    gen = SyntheticDataGenerator(seed=42)
    skus = gen.generate_skus(10)
    
    required_cols = ['sku_id', 'name', 'volume_m3_per_unit', 'weight_kg_per_unit']
    assert all(col in skus.columns for col in required_cols)
    assert len(skus) == 10


def test_sales_history_has_sales():
    """Test sales history contains actual sales."""
    gen = SyntheticDataGenerator(seed=42)
    warehouses = gen.generate_warehouses(2)
    skus = gen.generate_skus(5)
    start_date = datetime.now() - timedelta(days=30)
    
    sales = gen.generate_sales_history(warehouses, skus, start_date, 30)
    
    assert len(sales) > 0
    assert 'quantity_sold' in sales.columns
    assert sales['quantity_sold'].sum() > 0


def test_fleet_has_trucks():
    """Test fleet generation."""
    gen = SyntheticDataGenerator(seed=42)
    fleet = gen.generate_fleet()
    
    assert len(fleet) > 0
    assert 'truck_id' in fleet.columns
    assert 'volume_capacity_m3' in fleet.columns


def test_inventory_snapshot_structure():
    """Test inventory snapshot structure."""
    gen = SyntheticDataGenerator(seed=42)
    warehouses = gen.generate_warehouses(2)
    skus = gen.generate_skus(5)
    snapshot_date = datetime.now()
    
    inventory = gen.generate_inventory_snapshot(warehouses, skus, snapshot_date)
    
    assert len(inventory) == len(warehouses) * len(skus)
    assert 'quantity_on_hand' in inventory.columns
    assert 'quantity_available' in inventory.columns

