"""
Transfer simulator with stochastic lead-times and demand.

Simulates execution of replenishment plan and computes realized costs,
service levels, and waste due to perishability.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

from src.utils.cost_model import CostModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransferSimulator:
    """Simulate transfer execution with stochastic delays."""
    
    def __init__(self, plan_path: Path, warehouses: pd.DataFrame, skus: pd.DataFrame,
                 fleet: pd.DataFrame, sales_history: Optional[pd.DataFrame] = None,
                 seed: int = 42):
        """Initialize simulator."""
        self.plan = pd.read_csv(plan_path)
        self.warehouses = warehouses
        self.skus = skus
        self.fleet = fleet
        self.sales_history = sales_history
        
        self.cost_model = CostModel(warehouses, skus, fleet)
        
        np.random.seed(seed)
        self.seed = seed
        
    def simulate_lead_time(self, expected_lead_time: int, base_variance: float = 0.2) -> int:
        """Simulate stochastic lead time."""
        # Add random delay (normal distribution)
        delay = np.random.normal(0, expected_lead_time * base_variance)
        actual_lead_time = max(1, int(expected_lead_time + delay))
        return actual_lead_time
    
    def simulate_demand(self, forecast: float, variance: float = 0.3) -> float:
        """Simulate stochastic demand."""
        # Log-normal distribution around forecast
        actual_demand = np.random.lognormal(
            np.log(max(forecast, 1)), variance
        )
        return actual_demand
    
    def simulate_transfer(self, transfer_row: Dict, day: datetime) -> Dict:
        """Simulate a single transfer execution."""
        from_wh = transfer_row['from_wh']
        to_wh = transfer_row['to_wh']
        sku_id = transfer_row['sku_id']
        qty = transfer_row['qty']
        expected_arrival = pd.to_datetime(transfer_row['expected_arrival']).date()
        
        # Get expected lead time
        wh_data = self.warehouses[self.warehouses['warehouse_id'] == to_wh].iloc[0]
        expected_lead_time = wh_data['lead_time_days']
        
        # Simulate actual lead time
        actual_lead_time = self.simulate_lead_time(expected_lead_time)
        actual_arrival = day + timedelta(days=actual_lead_time)
        
        # Check perishability
        sku_data = self.skus[self.skus['sku_id'] == sku_id].iloc[0]
        waste_qty = 0
        if pd.notna(sku_data.get('perishability_ttl_days')):
            ttl = sku_data['perishability_ttl_days']
            if actual_lead_time > ttl:
                waste_qty = qty  # All items expire
            elif actual_lead_time > ttl * 0.8:
                waste_ratio = (actual_lead_time - ttl * 0.8) / (ttl * 0.2)
                waste_qty = int(qty * waste_ratio)
        
        # Calculate costs
        transport_cost = transfer_row.get('estimated_cost', 0)
        waste_cost = waste_qty * sku_data['unit_cost']
        
        return {
            'transfer_id': transfer_row.get('plan_id', 'UNKNOWN'),
            'from_wh': from_wh,
            'to_wh': to_wh,
            'sku_id': sku_id,
            'planned_qty': qty,
            'actual_qty_received': qty - waste_qty,
            'waste_qty': waste_qty,
            'planned_arrival': expected_arrival,
            'actual_arrival': actual_arrival,
            'planned_lead_time': expected_lead_time,
            'actual_lead_time': actual_lead_time,
            'transport_cost': transport_cost,
            'waste_cost': waste_cost,
            'total_cost': transport_cost + waste_cost,
        }
    
    def simulate_demand_and_stockouts(self, inventory_snapshot: pd.DataFrame,
                                     forecasts: pd.DataFrame,
                                     transfers: List[Dict]) -> Dict:
        """Simulate demand and calculate stockouts."""
        # Group transfers by warehouse and SKU
        transfers_by_wh_sku = {}
        for t in transfers:
            key = (t['to_wh'], t['sku_id'])
            if key not in transfers_by_wh_sku:
                transfers_by_wh_sku[key] = []
            transfers_by_wh_sku[key].append(t)
        
        stockouts = []
        total_demand = 0
        total_fulfilled = 0
        
        # For each warehouse-SKU, simulate demand
        for (warehouse_id, sku_id), transfer_list in transfers_by_wh_sku.items():
            # Get initial inventory
            inv = inventory_snapshot[
                (inventory_snapshot['warehouse_id'] == warehouse_id) &
                (inventory_snapshot['sku_id'] == sku_id)
            ]
            current_stock = inv['quantity_available'].iloc[0] if len(inv) > 0 else 0
            
            # Get forecast
            forecast_data = forecasts[
                (forecasts['warehouse_id'] == warehouse_id) &
                (forecasts['sku_id'] == sku_id)
            ]
            
            total_forecast = forecast_data['forecast'].sum() if len(forecast_data) > 0 else 0
            
            # Simulate actual demand
            actual_demand = self.simulate_demand(total_forecast)
            total_demand += actual_demand
            
            # Calculate incoming stock from transfers
            incoming_stock = sum(t['actual_qty_received'] for t in transfer_list)
            
            # Calculate available stock
            available_stock = current_stock + incoming_stock
            
            # Calculate stockout
            stockout_qty = max(0, actual_demand - available_stock)
            fulfilled_qty = min(actual_demand, available_stock)
            total_fulfilled += fulfilled_qty
            
            if stockout_qty > 0:
                stockouts.append({
                    'warehouse_id': warehouse_id,
                    'sku_id': sku_id,
                    'demand': actual_demand,
                    'available_stock': available_stock,
                    'stockout_qty': stockout_qty,
                    'service_level': fulfilled_qty / actual_demand if actual_demand > 0 else 1.0,
                })
        
        overall_service_level = total_fulfilled / total_demand if total_demand > 0 else 1.0
        
        return {
            'stockouts': pd.DataFrame(stockouts),
            'total_demand': total_demand,
            'total_fulfilled': total_fulfilled,
            'overall_service_level': overall_service_level,
        }
    
    def simulate(self, inventory_snapshot: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
        """Run complete simulation."""
        logger.info(f"Simulating {len(self.plan)} transfers...")
        
        simulated_transfers = []
        start_date = datetime.now()
        
        for _, row in self.plan.iterrows():
            ship_date = pd.to_datetime(row['ship_date']).date()
            sim_result = self.simulate_transfer(row.to_dict(), ship_date)
            simulated_transfers.append(sim_result)
        
        # Simulate demand and stockouts
        demand_results = self.simulate_demand_and_stockouts(
            inventory_snapshot, forecasts, simulated_transfers
        )
        
        # Create summary report
        transfers_df = pd.DataFrame(simulated_transfers)
        
        summary = {
            'simulation_date': datetime.now().date(),
            'plan_id': self.plan['plan_id'].iloc[0] if len(self.plan) > 0 else 'UNKNOWN',
            'total_transfers': len(simulated_transfers),
            'total_transport_cost': transfers_df['transport_cost'].sum(),
            'total_waste_cost': transfers_df['waste_cost'].sum(),
            'total_cost': transfers_df['total_cost'].sum(),
            'total_waste_units': transfers_df['waste_qty'].sum(),
            'overall_service_level': demand_results['overall_service_level'],
            'total_stockouts': len(demand_results['stockouts']),
            'total_demand': demand_results['total_demand'],
            'total_fulfilled': demand_results['total_fulfilled'],
        }
        
        # Combine results
        report = {
            'summary': summary,
            'transfers': transfers_df,
            'stockouts': demand_results['stockouts'],
        }
        
        return report
    
    def generate_report(self, inventory_snapshot: pd.DataFrame, 
                       forecasts: pd.DataFrame) -> Dict:
        """Generate simulation report."""
        results = self.simulate(inventory_snapshot, forecasts)
        return results


def main():
    parser = argparse.ArgumentParser(description='Simulate transfer execution')
    parser.add_argument('--plan', type=str, required=True,
                      help='Path to replenishment plan CSV')
    parser.add_argument('--out', type=str, required=True,
                      help='Output path for simulation report CSV')
    parser.add_argument('--warehouses', type=str, default='data/processed/warehouses.csv',
                      help='Path to warehouses CSV')
    parser.add_argument('--skus', type=str, default='data/processed/skus.csv',
                      help='Path to SKUs CSV')
    parser.add_argument('--fleet', type=str, default='data/raw/fleet.csv',
                      help='Path to fleet CSV')
    parser.add_argument('--inventory', type=str, default='data/processed/inventory_snapshot.csv',
                      help='Path to inventory snapshot CSV')
    parser.add_argument('--forecasts', type=str, default='data/forecasts/forecasts.csv',
                      help='Path to forecasts CSV')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Load data
    warehouses = pd.read_csv(args.warehouses)
    skus = pd.read_csv(args.skus)
    fleet = pd.read_csv(args.fleet) if Path(args.fleet).exists() else pd.DataFrame()
    inventory = pd.read_csv(args.inventory)
    forecasts = pd.read_csv(args.forecasts)
    
    # Create simulator
    simulator = TransferSimulator(
        Path(args.plan),
        warehouses,
        skus,
        fleet,
        seed=args.seed
    )
    
    # Run simulation
    report = simulator.generate_report(inventory, forecasts)
    
    # Save transfers DataFrame
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if 'transfers' in report and len(report['transfers']) > 0:
        report['transfers'].to_csv(out_path, index=False)
    else:
        # Create empty report if no transfers
        pd.DataFrame().to_csv(out_path, index=False)
    
    logger.info(f"Simulation report saved to {out_path}")


if __name__ == '__main__':
    main()

