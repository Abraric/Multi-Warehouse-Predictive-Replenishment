"""
Replenishment optimization engine using PuLP/OR-Tools.

Solves multi-warehouse replenishment planning with transport constraints,
perishability, truck capacity, driver availability, and dock windows.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
import uuid

try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False
    logging.warning("PuLP not available, using greedy heuristic")

from src.utils.cost_model import CostModel
from src.utils.explanations import ExplanationGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReplenishmentOptimizer:
    """Optimize replenishment plans across warehouses."""
    
    def __init__(self, snapshot_path: Path, forecasts_path: Path, 
                 warehouses: pd.DataFrame, skus: pd.DataFrame, fleet: pd.DataFrame,
                 inbound_pos: Optional[pd.DataFrame] = None):
        """Initialize optimizer with data."""
        self.inventory_snapshot = pd.read_csv(snapshot_path)
        self.forecasts = pd.read_csv(forecasts_path)
        self.warehouses = warehouses
        self.skus = skus
        self.fleet = fleet
        self.inbound_pos = inbound_pos if inbound_pos is not None else pd.DataFrame()
        
        self.cost_model = CostModel(warehouses, skus, fleet)
        self.explainer = ExplanationGenerator(warehouses, skus, self.forecasts)
        
        self.plan_id = f"PLAN-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        
    def optimize_with_pulp(self, horizon_days: int, start_date: datetime) -> pd.DataFrame:
        """Optimize using PuLP solver."""
        if not PULP_AVAILABLE:
            logger.warning("PuLP not available, falling back to greedy heuristic")
            return self.greedy_heuristic(horizon_days, start_date)
        
        logger.info(f"Starting PuLP optimization for {horizon_days} days...")
        
        # Create problem
        prob = pulp.LpProblem("Replenishment_Optimization", pulp.LpMinimize)
        
        # Decision variables: transfer[from_wh, to_wh, sku_id, day] = quantity
        transfers = {}
        dates = [start_date + timedelta(days=i) for i in range(horizon_days)]
        
        warehouse_ids = self.warehouses['warehouse_id'].unique()
        sku_ids = self.skus['sku_id'].unique()
        
        # Create transfer variables
        for from_wh in warehouse_ids:
            for to_wh in warehouse_ids:
                if from_wh == to_wh:
                    continue
                for sku_id in sku_ids:
                    for day_idx, date in enumerate(dates):
                        var_name = f"transfer_{from_wh}_{to_wh}_{sku_id}_{day_idx}"
                        transfers[(from_wh, to_wh, sku_id, day_idx)] = pulp.LpVariable(
                            var_name, lowBound=0, cat='Integer'
                        )
        
        # Objective: minimize total cost
        objective = []
        
        for (from_wh, to_wh, sku_id, day_idx), var in transfers.items():
            date = dates[day_idx]
            # Transport cost
            distance = self.cost_model.calculate_distance(from_wh, to_wh)
            truck_cost = distance * 1.5  # Simplified
            transport_cost = truck_cost * var
            
            # Perishability cost
            sku_data = self.skus[self.skus['sku_id'] == sku_id].iloc[0]
            lead_time = self.warehouses[self.warehouses['warehouse_id'] == to_wh]['lead_time_days'].iloc[0]
            if pd.notna(sku_data.get('perishability_ttl_days')):
                ttl = sku_data['perishability_ttl_days']
                if lead_time > ttl:
                    # Heavy penalty
                    waste_cost = var * sku_data['unit_cost'] * 10
                else:
                    waste_cost = 0
            else:
                waste_cost = 0
            
            objective.append(transport_cost + waste_cost)
        
        # Stockout penalty
        for to_wh in warehouse_ids:
            for sku_id in sku_ids:
                # Get current inventory
                inv = self.inventory_snapshot[
                    (self.inventory_snapshot['warehouse_id'] == to_wh) &
                    (self.inventory_snapshot['sku_id'] == sku_id)
                ]
                current_stock = inv['quantity_available'].iloc[0] if len(inv) > 0 else 0
                
                # Get forecasted demand
                forecast = self.forecasts[
                    (self.forecasts['warehouse_id'] == to_wh) &
                    (self.forecasts['sku_id'] == sku_id)
                ]
                total_demand = forecast['forecast'].sum() if len(forecast) > 0 else 0
                
                # Incoming transfers
                incoming = pulp.lpSum([
                    transfers.get((from_wh, to_wh, sku_id, day_idx), 0)
                    for from_wh in warehouse_ids
                    for day_idx in range(horizon_days)
                    if from_wh != to_wh
                ])
                
                # Stockout variable
                stockout = pulp.LpVariable(f"stockout_{to_wh}_{sku_id}", lowBound=0)
                
                # Constraint: stockout = max(0, demand - stock - incoming)
                prob += stockout >= total_demand - current_stock - incoming
                
                # Add stockout cost to objective
                wh_data = self.warehouses[self.warehouses['warehouse_id'] == to_wh].iloc[0]
                stockout_cost = stockout * wh_data['stockout_cost_per_unit']
                objective.append(stockout_cost)
        
        prob += pulp.lpSum(objective)
        
        # Constraints
        
        # 1. Warehouse capacity constraints
        for wh in warehouse_ids:
            for day_idx in range(horizon_days):
                # Total volume in warehouse
                total_volume = pulp.lpSum([
                    transfers.get((from_wh, wh, sku_id, day_idx), 0) * 
                    self.skus[self.skus['sku_id'] == sku_id]['volume_m3_per_unit'].iloc[0]
                    for from_wh in warehouse_ids
                    for sku_id in sku_ids
                    if from_wh != wh
                ])
                
                wh_capacity = self.warehouses[self.warehouses['warehouse_id'] == wh]['capacity_volume_m3'].iloc[0]
                prob += total_volume <= wh_capacity
        
        # 2. Perishability TTL constraints
        for (from_wh, to_wh, sku_id, day_idx), var in transfers.items():
            sku_data = self.skus[self.skus['sku_id'] == sku_id].iloc[0]
            if pd.notna(sku_data.get('perishability_ttl_days')):
                ttl = sku_data['perishability_ttl_days']
                lead_time = self.warehouses[self.warehouses['warehouse_id'] == to_wh]['lead_time_days'].iloc[0]
                
                if lead_time > ttl:
                    # Disallow transfer
                    prob += var == 0
        
        # 3. Source warehouse inventory constraints
        for from_wh in warehouse_ids:
            for sku_id in sku_ids:
                inv = self.inventory_snapshot[
                    (self.inventory_snapshot['warehouse_id'] == from_wh) &
                    (self.inventory_snapshot['sku_id'] == sku_id)
                ]
                available_stock = inv['quantity_available'].iloc[0] if len(inv) > 0 else 0
                
                # Total outgoing transfers
                outgoing = pulp.lpSum([
                    transfers.get((from_wh, to_wh, sku_id, day_idx), 0)
                    for to_wh in warehouse_ids
                    for day_idx in range(horizon_days)
                    if to_wh != from_wh
                ])
                
                prob += outgoing <= available_stock
        
        # Solve
        logger.info("Solving optimization problem...")
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if prob.status != pulp.LpStatusOptimal:
            logger.warning(f"Optimization status: {pulp.LpStatus[prob.status]}, falling back to greedy")
            return self.greedy_heuristic(horizon_days, start_date)
        
        # Extract solution
        plan_rows = []
        for (from_wh, to_wh, sku_id, day_idx), var in transfers.items():
            qty = pulp.value(var)
            if qty and qty > 0:
                date = dates[day_idx]
                arrival_date = date + timedelta(days=self.warehouses[
                    self.warehouses['warehouse_id'] == to_wh
                ]['lead_time_days'].iloc[0])
                
                # Calculate cost
                cost = self.cost_model.transport_cost(from_wh, to_wh, int(qty), sku_id)
                
                plan_rows.append({
                    'plan_id': self.plan_id,
                    'run_date': start_date.date(),
                    'from_wh': from_wh,
                    'to_wh': to_wh,
                    'sku_id': sku_id,
                    'qty': int(qty),
                    'ship_date': date.date(),
                    'expected_arrival': arrival_date.date(),
                    'truck_type': 'Medium',  # Simplified
                    'driver_id': None,
                    'estimated_cost': cost,
                    'reason_code': 'OPTIMIZATION',
                })
        
        return pd.DataFrame(plan_rows)
    
    def greedy_heuristic(self, horizon_days: int, start_date: datetime) -> pd.DataFrame:
        """Greedy heuristic fallback when optimization solver unavailable."""
        logger.info("Using greedy heuristic...")
        
        plan_rows = []
        dates = [start_date + timedelta(days=i) for i in range(horizon_days)]
        
        # For each warehouse-SKU combination, check if replenishment needed
        for _, wh in self.warehouses.iterrows():
            warehouse_id = wh['warehouse_id']
            
            for _, sku in self.skus.iterrows():
                sku_id = sku['sku_id']
                
                # Get current inventory
                inv = self.inventory_snapshot[
                    (self.inventory_snapshot['warehouse_id'] == warehouse_id) &
                    (self.inventory_snapshot['sku_id'] == sku_id)
                ]
                current_stock = inv['quantity_available'].iloc[0] if len(inv) > 0 else 0
                
                # Get forecast
                forecast = self.forecasts[
                    (self.forecasts['warehouse_id'] == warehouse_id) &
                    (self.forecasts['sku_id'] == sku_id)
                ]
                total_demand = forecast['forecast'].sum() if len(forecast) > 0 else 0
                
                # Calculate safety stock
                safety_stock_days = sku.get('safety_stock_days', 7)
                safety_stock = total_demand * safety_stock_days / horizon_days
                
                # Check if replenishment needed
                if current_stock < safety_stock:
                    needed_qty = int(safety_stock - current_stock + total_demand)
                    
                    # Find source warehouse with available stock
                    for _, source_wh in self.warehouses.iterrows():
                        if source_wh['warehouse_id'] == warehouse_id:
                            continue
                        
                        source_inv = self.inventory_snapshot[
                            (self.inventory_snapshot['warehouse_id'] == source_wh['warehouse_id']) &
                            (self.inventory_snapshot['sku_id'] == sku_id)
                        ]
                        source_stock = source_inv['quantity_available'].iloc[0] if len(source_inv) > 0 else 0
                        
                        if source_stock >= needed_qty:
                            # Check perishability constraint
                            if pd.notna(sku.get('perishability_ttl_days')):
                                ttl = sku['perishability_ttl_days']
                                lead_time = wh['lead_time_days']
                                if lead_time > ttl:
                                    continue  # Skip if violates TTL
                            
                            # Create transfer
                            ship_date = dates[0]
                            arrival_date = ship_date + timedelta(days=int(wh['lead_time_days']))
                            cost = self.cost_model.transport_cost(
                                source_wh['warehouse_id'], warehouse_id, needed_qty, sku_id
                            )
                            
                            plan_rows.append({
                                'plan_id': self.plan_id,
                                'run_date': start_date.date(),
                                'from_wh': source_wh['warehouse_id'],
                                'to_wh': warehouse_id,
                                'sku_id': sku_id,
                                'qty': needed_qty,
                                'ship_date': ship_date.date(),
                                'expected_arrival': arrival_date.date(),
                                'truck_type': 'Refrigerated' if sku.get('requires_refrigerated') else 'Medium',
                                'driver_id': None,
                                'estimated_cost': cost,
                                'reason_code': 'GREEDY_HEURISTIC',
                            })
                            break
        
        return pd.DataFrame(plan_rows)
    
    def add_explanations(self, plan: pd.DataFrame) -> pd.DataFrame:
        """Add explanations and cost deltas to plan."""
        plan = plan.copy()
        
        explanations = []
        cost_deltas = []
        top_drivers_list = []
        
        for _, row in plan.iterrows():
            exp = self.explainer.explain_transfer(
                row.to_dict(), self.inventory_snapshot, self.forecasts
            )
            explanations.append(exp)
            
            drivers = self.explainer.get_top_drivers(
                row.to_dict(), self.inventory_snapshot, self.forecasts
            )
            top_drivers_list.append('; '.join(drivers))
            
            delta = self.explainer.calculate_delta_cost(row.to_dict(), self.cost_model)
            cost_deltas.append(delta)
        
        plan['explanation_text'] = explanations
        plan['delta_cost_if_not_transferred'] = cost_deltas
        plan['top_drivers'] = top_drivers_list
        
        return plan
    
    def optimize(self, horizon_days: int, start_date: Optional[datetime] = None) -> pd.DataFrame:
        """Main optimization method."""
        if start_date is None:
            start_date = datetime.now()
        
        # Run optimization
        if PULP_AVAILABLE:
            plan = self.optimize_with_pulp(horizon_days, start_date)
        else:
            plan = self.greedy_heuristic(horizon_days, start_date)
        
        # Add explanations
        plan = self.add_explanations(plan)
        
        logger.info(f"Generated plan with {len(plan)} transfers")
        return plan


def main():
    parser = argparse.ArgumentParser(description='Optimize replenishment plan')
    parser.add_argument('--snapshot', type=str, required=True,
                      help='Path to inventory snapshot CSV')
    parser.add_argument('--forecasts', type=str, required=True,
                      help='Path to forecasts directory or CSV')
    parser.add_argument('--horizon', type=int, default=14,
                      help='Planning horizon in days')
    parser.add_argument('--out', type=str, required=True,
                      help='Output path for plan CSV')
    parser.add_argument('--warehouses', type=str, default='data/processed/warehouses.csv',
                      help='Path to warehouses CSV')
    parser.add_argument('--skus', type=str, default='data/processed/skus.csv',
                      help='Path to SKUs CSV')
    parser.add_argument('--fleet', type=str, default='data/raw/fleet.csv',
                      help='Path to fleet CSV')
    
    args = parser.parse_args()
    
    # Load data
    warehouses = pd.read_csv(args.warehouses)
    skus = pd.read_csv(args.skus)
    fleet = pd.read_csv(args.fleet) if Path(args.fleet).exists() else pd.DataFrame()
    
    # Load forecasts
    if Path(args.forecasts).is_dir():
        forecast_path = Path(args.forecasts) / 'forecasts.csv'
    else:
        forecast_path = Path(args.forecasts)
    
    # Create optimizer
    optimizer = ReplenishmentOptimizer(
        Path(args.snapshot),
        forecast_path,
        warehouses,
        skus,
        fleet
    )
    
    # Optimize
    plan = optimizer.optimize(args.horizon)
    
    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(out_path, index=False)
    
    logger.info(f"Plan saved to {out_path}")


if __name__ == '__main__':
    main()

