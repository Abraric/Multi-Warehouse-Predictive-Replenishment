"""
Synthetic data generator for multi-warehouse inventory and sales data.

Generates realistic datasets including warehouses, SKUs, sales history, fleet,
drivers, and operational constraints.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """Generate synthetic multi-warehouse inventory and sales data."""
    
    def __init__(self, seed: int = 42):
        """Initialize generator with seed for reproducibility."""
        np.random.seed(seed)
        self.seed = seed
        
    def generate_warehouses(self, n_warehouses: int = 5) -> pd.DataFrame:
        """Generate warehouse metadata."""
        warehouses = []
        cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 
                 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose']
        
        # Base coordinates (simplified)
        base_coords = {
            'New York': (40.7128, -74.0060),
            'Los Angeles': (34.0522, -118.2437),
            'Chicago': (41.8781, -87.6298),
            'Houston': (29.7604, -95.3698),
            'Phoenix': (33.4484, -112.0740),
        }
        
        for i in range(n_warehouses):
            city = cities[i % len(cities)]
            lat, lon = base_coords.get(city, (40.0 + i*2, -100.0 + i*3))
            
            warehouses.append({
                'warehouse_id': f'WH-{i+1:03d}',
                'name': f'{city} Distribution Center',
                'city': city,
                'latitude': lat + np.random.uniform(-0.5, 0.5),
                'longitude': lon + np.random.uniform(-0.5, 0.5),
                'capacity_volume_m3': np.random.uniform(5000, 20000),
                'capacity_weight_kg': np.random.uniform(100000, 500000),
                'lead_time_days': np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2]),
                'transfer_cost_per_km': np.random.uniform(0.5, 2.0),
                'holding_cost_per_unit_per_day': np.random.uniform(0.01, 0.05),
                'stockout_cost_per_unit': np.random.uniform(10, 50),
                'num_docks': np.random.randint(2, 8),
                'dock_window_start': '08:00',
                'dock_window_end': '18:00',
            })
        
        return pd.DataFrame(warehouses)
    
    def generate_skus(self, n_skus: int = 50) -> pd.DataFrame:
        """Generate SKU metadata."""
        skus = []
        categories = ['Electronics', 'Clothing', 'Food', 'Furniture', 'Toys', 
                     'Books', 'Sports', 'Beauty', 'Automotive', 'Home']
        
        for i in range(n_skus):
            category = categories[i % len(categories)]
            is_perishable = category == 'Food' and np.random.random() < 0.7
            
            skus.append({
                'sku_id': f'SKU-{i+1:04d}',
                'name': f'{category} Product {i+1}',
                'category': category,
                'volume_m3_per_unit': np.random.uniform(0.001, 0.1),
                'weight_kg_per_unit': np.random.uniform(0.1, 10.0),
                'unit_cost': np.random.uniform(5, 500),
                'perishability_ttl_days': np.random.randint(1, 30) if is_perishable else None,
                'requires_refrigerated': is_perishable and np.random.random() < 0.5,
                'safety_stock_days': np.random.randint(3, 14),
            })
        
        return pd.DataFrame(skus)
    
    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate approximate distance in km using Haversine formula."""
        R = 6371  # Earth radius in km
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = (np.sin(dlat/2)**2 + 
             np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c
    
    def generate_sales_history(self, warehouses: pd.DataFrame, skus: pd.DataFrame,
                              start_date: datetime, days: int) -> pd.DataFrame:
        """Generate historical sales data with seasonality and promotions."""
        sales = []
        dates = [start_date + timedelta(days=i) for i in range(days)]
        
        # Base demand per SKU (varies by category)
        base_demand = {}
        for _, sku in skus.iterrows():
            if sku['category'] == 'Electronics':
                base_demand[sku['sku_id']] = np.random.uniform(20, 100)
            elif sku['category'] == 'Food':
                base_demand[sku['sku_id']] = np.random.uniform(50, 200)
            else:
                base_demand[sku['sku_id']] = np.random.uniform(10, 80)
        
        # Generate promotions (spikes)
        promo_dates = {}
        for sku_id in base_demand.keys():
            promo_dates[sku_id] = np.random.choice(dates, size=np.random.randint(2, 5), replace=False)
        
        for date in dates:
            day_of_week = date.weekday()
            is_weekend = day_of_week >= 5
            is_holiday = date.month == 12 and date.day >= 20  # Holiday season
            
            for _, warehouse in warehouses.iterrows():
                for _, sku in skus.iterrows():
                    base = base_demand[sku['sku_id']]
                    
                    # Seasonality
                    seasonal_factor = 1.0
                    if date.month in [11, 12]:  # Holiday season
                        seasonal_factor = 1.5
                    elif date.month in [6, 7, 8]:  # Summer
                        seasonal_factor = 1.2
                    
                    # Weekend effect
                    if is_weekend:
                        seasonal_factor *= 1.1
                    
                    # Promotion spike
                    promo_factor = 2.5 if date in promo_dates.get(sku['sku_id'], []) else 1.0
                    
                    # Random variation
                    noise = np.random.lognormal(0, 0.3)
                    
                    # Warehouse-specific demand variation
                    wh_factor = np.random.uniform(0.8, 1.2)
                    
                    demand = max(0, int(base * seasonal_factor * promo_factor * noise * wh_factor))
                    
                    if demand > 0:
                        sales.append({
                            'date': date.date(),
                            'warehouse_id': warehouse['warehouse_id'],
                            'sku_id': sku['sku_id'],
                            'quantity_sold': demand,
                            'revenue': demand * sku['unit_cost'] * np.random.uniform(0.9, 1.1),
                        })
        
        return pd.DataFrame(sales)
    
    def generate_fleet(self) -> pd.DataFrame:
        """Generate fleet metadata (trucks)."""
        truck_types = [
            {'type': 'Small', 'volume_m3': 20, 'weight_kg': 5000, 'cost_per_km': 1.0, 'refrigerated': False},
            {'type': 'Medium', 'volume_m3': 50, 'weight_kg': 15000, 'cost_per_km': 1.5, 'refrigerated': False},
            {'type': 'Large', 'volume_m3': 100, 'weight_kg': 30000, 'cost_per_km': 2.0, 'refrigerated': False},
            {'type': 'Refrigerated', 'volume_m3': 40, 'weight_kg': 12000, 'cost_per_km': 2.5, 'refrigerated': True},
        ]
        
        fleet = []
        for i, truck_type in enumerate(truck_types):
            for j in range(np.random.randint(3, 8)):
                fleet.append({
                    'truck_id': f'TRUCK-{truck_type["type"][:1]}-{j+1:03d}',
                    'truck_type': truck_type['type'],
                    'volume_capacity_m3': truck_type['volume_m3'],
                    'weight_capacity_kg': truck_type['weight_kg'],
                    'cost_per_km': truck_type['cost_per_km'],
                    'is_refrigerated': truck_type['refrigerated'],
                    'max_daily_hours': 10,
                    'fixed_cost_per_trip': np.random.uniform(50, 200),
                })
        
        return pd.DataFrame(fleet)
    
    def generate_drivers(self, n_drivers: int = 20) -> pd.DataFrame:
        """Generate driver availability data."""
        drivers = []
        shift_types = [
            {'start': '06:00', 'end': '14:00'},
            {'start': '08:00', 'end': '16:00'},
            {'start': '10:00', 'end': '18:00'},
            {'start': '14:00', 'end': '22:00'},
        ]
        
        for i in range(n_drivers):
            shift = shift_types[i % len(shift_types)]
            drivers.append({
                'driver_id': f'DRIVER-{i+1:03d}',
                'name': f'Driver {i+1}',
                'shift_start': shift['start'],
                'shift_end': shift['end'],
                'max_hours_per_day': np.random.uniform(8, 10),
                'can_drive_refrigerated': np.random.random() < 0.3,
            })
        
        return pd.DataFrame(drivers)
    
    def generate_inbound_pos(self, warehouses: pd.DataFrame, skus: pd.DataFrame,
                            start_date: datetime, days: int) -> pd.DataFrame:
        """Generate inbound purchase orders from suppliers."""
        pos = []
        
        # Generate PO schedule
        for i in range(0, days, 7):  # Weekly POs
            po_date = start_date + timedelta(days=i)
            arrival_date = po_date + timedelta(days=np.random.randint(3, 10))
            
            for _, warehouse in warehouses.iterrows():
                # Select random SKUs for this PO
                n_skus_in_po = np.random.randint(5, 15)
                selected_skus = skus.sample(n=min(n_skus_in_po, len(skus)))
                
                for _, sku in selected_skus.iterrows():
                    quantity = np.random.randint(100, 1000)
                    # Some partial deliveries
                    if np.random.random() < 0.2:
                        quantity = int(quantity * np.random.uniform(0.5, 0.9))
                    
                    pos.append({
                        'po_id': f'PO-{warehouse["warehouse_id"]}-{i:04d}',
                        'warehouse_id': warehouse['warehouse_id'],
                        'sku_id': sku['sku_id'],
                        'order_date': po_date.date(),
                        'expected_arrival_date': arrival_date.date(),
                        'quantity_ordered': quantity,
                        'quantity_received': None,  # Will be filled when received
                        'supplier_lead_time_days': (arrival_date - po_date).days,
                        'status': 'pending',
                    })
        
        return pd.DataFrame(pos)
    
    def generate_inventory_snapshot(self, warehouses: pd.DataFrame, skus: pd.DataFrame,
                                   snapshot_date: datetime) -> pd.DataFrame:
        """Generate current inventory snapshot."""
        inventory = []
        
        for _, warehouse in warehouses.iterrows():
            for _, sku in skus.iterrows():
                # Generate realistic inventory levels
                base_stock = np.random.uniform(100, 2000)
                # Some SKUs might be low/out of stock
                if np.random.random() < 0.1:
                    base_stock = np.random.uniform(0, 50)
                
                inventory.append({
                    'snapshot_date': snapshot_date.date(),
                    'warehouse_id': warehouse['warehouse_id'],
                    'sku_id': sku['sku_id'],
                    'quantity_on_hand': max(0, int(base_stock)),
                    'quantity_reserved': 0,
                    'quantity_available': max(0, int(base_stock)),
                })
        
        return pd.DataFrame(inventory)
    
    def generate_all(self, n_warehouses: int, n_skus: int, days: int,
                     start_date: datetime = None) -> Dict[str, pd.DataFrame]:
        """Generate all synthetic datasets."""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=days)
        
        logger.info(f"Generating synthetic data: {n_warehouses} warehouses, {n_skus} SKUs, {days} days")
        
        warehouses = self.generate_warehouses(n_warehouses)
        skus = self.generate_skus(n_skus)
        sales = self.generate_sales_history(warehouses, skus, start_date, days)
        fleet = self.generate_fleet()
        drivers = self.generate_drivers()
        inbound_pos = self.generate_inbound_pos(warehouses, skus, start_date, days)
        inventory = self.generate_inventory_snapshot(warehouses, skus, start_date + timedelta(days=days))
        
        return {
            'warehouses': warehouses,
            'skus': skus,
            'sales_history': sales,
            'fleet': fleet,
            'drivers': drivers,
            'inbound_pos': inbound_pos,
            'inventory_snapshot': inventory,
        }
    
    def save_to_csv(self, data: Dict[str, pd.DataFrame], out_dir: Path):
        """Save all datasets to CSV files."""
        out_dir.mkdir(parents=True, exist_ok=True)
        
        for name, df in data.items():
            filepath = out_dir / f'{name}.csv'
            df.to_csv(filepath, index=False)
            logger.info(f"Saved {name} to {filepath} ({len(df)} rows)")


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic multi-warehouse data')
    parser.add_argument('--out-dir', type=str, default='data/raw',
                      help='Output directory for CSV files')
    parser.add_argument('--warehouses', type=int, default=5,
                      help='Number of warehouses')
    parser.add_argument('--skus', type=int, default=50,
                      help='Number of SKUs')
    parser.add_argument('--days', type=int, default=180,
                      help='Number of days of historical data')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    generator = SyntheticDataGenerator(seed=args.seed)
    start_date = datetime.now() - timedelta(days=args.days)
    
    data = generator.generate_all(
        n_warehouses=args.warehouses,
        n_skus=args.skus,
        days=args.days,
        start_date=start_date
    )
    
    out_dir = Path(args.out_dir)
    generator.save_to_csv(data, out_dir)
    
    logger.info(f"Data generation complete. Files saved to {out_dir}")


if __name__ == '__main__':
    main()

