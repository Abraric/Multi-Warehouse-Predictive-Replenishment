"""
ETL pipeline for multi-warehouse inventory and sales data.

Loads raw CSVs, validates data, creates aggregations, and produces
cleaned tables suitable for forecasting and optimization.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ETLPipeline:
    """ETL pipeline for warehouse data processing."""
    
    def __init__(self):
        """Initialize ETL pipeline."""
        self.warehouses = None
        self.skus = None
        self.sales_history = None
        self.inventory_snapshot = None
        self.inbound_pos = None
        
    def load_raw_data(self, in_dir: Path) -> Dict[str, pd.DataFrame]:
        """Load all raw CSV files."""
        logger.info(f"Loading raw data from {in_dir}")
        
        data = {}
        files = {
            'warehouses': 'warehouses.csv',
            'skus': 'skus.csv',
            'sales_history': 'sales_history.csv',
            'inventory_snapshot': 'inventory_snapshot.csv',
            'inbound_pos': 'inbound_pos.csv',
        }
        
        for key, filename in files.items():
            filepath = in_dir / filename
            if filepath.exists():
                df = pd.read_csv(filepath)
                # Convert date columns
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date']).dt.date
                if 'snapshot_date' in df.columns:
                    df['snapshot_date'] = pd.to_datetime(df['snapshot_date']).dt.date
                if 'order_date' in df.columns:
                    df['order_date'] = pd.to_datetime(df['order_date']).dt.date
                if 'expected_arrival_date' in df.columns:
                    df['expected_arrival_date'] = pd.to_datetime(df['expected_arrival_date']).dt.date
                
                data[key] = df
                logger.info(f"Loaded {key}: {len(df)} rows")
            else:
                logger.warning(f"File not found: {filepath}")
        
        return data
    
    def validate_data(self, data: Dict[str, pd.DataFrame]) -> bool:
        """Validate referential integrity and data quality."""
        logger.info("Validating data integrity...")
        
        # Check required files exist
        required = ['warehouses', 'skus', 'sales_history', 'inventory_snapshot']
        for key in required:
            if key not in data:
                logger.error(f"Missing required data: {key}")
                return False
        
        # Validate referential integrity
        warehouse_ids = set(data['warehouses']['warehouse_id'].unique())
        sku_ids = set(data['skus']['sku_id'].unique())
        
        # Check sales history
        if 'sales_history' in data:
            invalid_wh = set(data['sales_history']['warehouse_id'].unique()) - warehouse_ids
            invalid_sku = set(data['sales_history']['sku_id'].unique()) - sku_ids
            if invalid_wh:
                logger.warning(f"Invalid warehouse IDs in sales_history: {invalid_wh}")
            if invalid_sku:
                logger.warning(f"Invalid SKU IDs in sales_history: {invalid_sku}")
        
        # Check inventory snapshot
        invalid_wh = set(data['inventory_snapshot']['warehouse_id'].unique()) - warehouse_ids
        invalid_sku = set(data['inventory_snapshot']['sku_id'].unique()) - sku_ids
        if invalid_wh:
            logger.warning(f"Invalid warehouse IDs in inventory_snapshot: {invalid_wh}")
        if invalid_sku:
            logger.warning(f"Invalid SKU IDs in inventory_snapshot: {invalid_sku}")
        
        logger.info("Data validation complete")
        return True
    
    def create_daily_aggregations(self, sales_history: pd.DataFrame) -> pd.DataFrame:
        """Create daily sales aggregations per SKU per warehouse."""
        logger.info("Creating daily sales aggregations...")
        
        daily_sales = sales_history.groupby(['date', 'warehouse_id', 'sku_id']).agg({
            'quantity_sold': 'sum',
            'revenue': 'sum',
        }).reset_index()
        
        daily_sales = daily_sales.sort_values(['warehouse_id', 'sku_id', 'date'])
        return daily_sales
    
    def create_rolling_features(self, daily_sales: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window features for forecasting."""
        logger.info("Creating rolling features...")
        
        features = []
        
        for (warehouse_id, sku_id), group in daily_sales.groupby(['warehouse_id', 'sku_id']):
            group = group.sort_values('date').copy()
            
            # Rolling windows
            group['rolling_7d_mean'] = group['quantity_sold'].rolling(window=7, min_periods=1).mean()
            group['rolling_7d_std'] = group['quantity_sold'].rolling(window=7, min_periods=1).std().fillna(0)
            group['rolling_28d_mean'] = group['quantity_sold'].rolling(window=28, min_periods=1).mean()
            group['rolling_28d_std'] = group['quantity_sold'].rolling(window=28, min_periods=1).std().fillna(0)
            
            # Lag features
            group['lag_1d'] = group['quantity_sold'].shift(1).fillna(0)
            group['lag_7d'] = group['quantity_sold'].shift(7).fillna(0)
            group['lag_28d'] = group['quantity_sold'].shift(28).fillna(0)
            
            # Date features
            group['day_of_week'] = pd.to_datetime(group['date']).dt.dayofweek
            group['day_of_month'] = pd.to_datetime(group['date']).dt.day
            group['month'] = pd.to_datetime(group['date']).dt.month
            group['is_weekend'] = (group['day_of_week'] >= 5).astype(int)
            
            # Seasonality flags
            group['is_holiday_season'] = (group['month'].isin([11, 12])).astype(int)
            group['is_summer'] = (group['month'].isin([6, 7, 8])).astype(int)
            
            # Promotion indicator (spike detection)
            mean_demand = group['quantity_sold'].mean()
            std_demand = group['quantity_sold'].std()
            group['is_promotion'] = (
                (group['quantity_sold'] > mean_demand + 2 * std_demand)
            ).astype(int)
            
            features.append(group)
        
        result = pd.concat(features, ignore_index=True)
        return result
    
    def create_inventory_features(self, inventory_snapshot: pd.DataFrame, 
                                  skus: pd.DataFrame) -> pd.DataFrame:
        """Enrich inventory snapshot with SKU features."""
        logger.info("Creating inventory features...")
        
        inventory = inventory_snapshot.merge(
            skus[['sku_id', 'safety_stock_days', 'perishability_ttl_days', 
                  'requires_refrigerated', 'unit_cost']],
            on='sku_id',
            how='left'
        )
        
        # Calculate days of supply (simplified)
        inventory['days_of_supply'] = inventory['quantity_available'] / 100  # Placeholder
        
        return inventory
    
    def process(self, in_dir: Path, out_dir: Path):
        """Run complete ETL pipeline."""
        logger.info("Starting ETL pipeline...")
        
        # Load raw data
        raw_data = self.load_raw_data(in_dir)
        
        if not raw_data:
            logger.error("No data loaded. Exiting.")
            return
        
        # Validate
        if not self.validate_data(raw_data):
            logger.error("Data validation failed. Exiting.")
            return
        
        # Create processed datasets
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Daily sales aggregations
        if 'sales_history' in raw_data:
            daily_sales = self.create_daily_aggregations(raw_data['sales_history'])
            daily_sales_with_features = self.create_rolling_features(daily_sales)
            daily_sales_with_features.to_csv(
                out_dir / 'daily_sales_aggregated.csv', index=False
            )
            logger.info(f"Saved daily sales aggregations: {len(daily_sales_with_features)} rows")
        
        # Inventory snapshot with features
        if 'inventory_snapshot' in raw_data and 'skus' in raw_data:
            inventory_features = self.create_inventory_features(
                raw_data['inventory_snapshot'],
                raw_data['skus']
            )
            inventory_features.to_csv(
                out_dir / 'inventory_snapshot.csv', index=False
            )
            logger.info(f"Saved inventory snapshot: {len(inventory_features)} rows")
        
        # Save cleaned versions of reference data
        if 'warehouses' in raw_data:
            raw_data['warehouses'].to_csv(out_dir / 'warehouses.csv', index=False)
        if 'skus' in raw_data:
            raw_data['skus'].to_csv(out_dir / 'skus.csv', index=False)
        if 'inbound_pos' in raw_data:
            raw_data['inbound_pos'].to_csv(out_dir / 'inbound_pos.csv', index=False)
        
        logger.info(f"ETL pipeline complete. Output saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description='ETL pipeline for warehouse data')
    parser.add_argument('--in-dir', type=str, default='data/raw',
                      help='Input directory with raw CSV files')
    parser.add_argument('--out-dir', type=str, default='data/processed',
                      help='Output directory for processed CSV files')
    
    args = parser.parse_args()
    
    pipeline = ETLPipeline()
    pipeline.process(Path(args.in_dir), Path(args.out_dir))


if __name__ == '__main__':
    main()

