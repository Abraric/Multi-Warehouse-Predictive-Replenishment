"""Tests for ETL pipeline."""

import pytest
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

from src.etl.etl import ETLPipeline
from src.data.synthetic_generator import SyntheticDataGenerator


@pytest.fixture
def sample_data(tmp_path):
    """Generate sample data for testing."""
    gen = SyntheticDataGenerator(seed=42)
    start_date = datetime.now() - timedelta(days=30)
    data = gen.generate_all(3, 10, 30, start_date)
    
    # Save to temp directory
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    gen.save_to_csv(data, raw_dir)
    
    return raw_dir


def test_etl_loads_data(sample_data):
    """Test ETL loads raw data correctly."""
    pipeline = ETLPipeline()
    data = pipeline.load_raw_data(sample_data)
    
    assert 'warehouses' in data
    assert 'skus' in data
    assert 'sales_history' in data
    assert len(data['warehouses']) > 0


def test_etl_creates_aggregations(sample_data, tmp_path):
    """Test ETL creates daily aggregations."""
    pipeline = ETLPipeline()
    out_dir = tmp_path / "processed"
    
    pipeline.process(sample_data, out_dir)
    
    # Check aggregated file exists
    agg_file = out_dir / 'daily_sales_aggregated.csv'
    assert agg_file.exists()
    
    df = pd.read_csv(agg_file)
    assert len(df) > 0
    assert 'rolling_7d_mean' in df.columns


def test_etl_creates_features(sample_data, tmp_path):
    """Test ETL creates rolling features."""
    pipeline = ETLPipeline()
    out_dir = tmp_path / "processed"
    
    pipeline.process(sample_data, out_dir)
    
    agg_file = out_dir / 'daily_sales_aggregated.csv'
    df = pd.read_csv(agg_file)
    
    # Check features exist
    assert 'rolling_7d_mean' in df.columns
    assert 'rolling_28d_mean' in df.columns
    assert 'lag_1d' in df.columns
    assert 'is_weekend' in df.columns

