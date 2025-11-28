"""
Generate human-readable explanations for replenishment decisions.
"""

from typing import Dict, List
import pandas as pd


class ExplanationGenerator:
    """Generate explanations for optimization decisions."""
    
    def __init__(self, warehouses: pd.DataFrame, skus: pd.DataFrame, forecasts: pd.DataFrame):
        """Initialize explanation generator."""
        self.warehouses = warehouses.set_index('warehouse_id')
        self.skus = skus.set_index('sku_id')
        self.forecasts = forecasts
    
    def explain_transfer(self, plan_row: Dict, inventory: pd.DataFrame, 
                        forecasts: pd.DataFrame) -> str:
        """Generate explanation for a transfer decision."""
        from_wh = plan_row['from_wh']
        to_wh = plan_row['to_wh']
        sku_id = plan_row['sku_id']
        qty = plan_row['qty']
        
        # Get current inventory
        to_inv = inventory[
            (inventory['warehouse_id'] == to_wh) & 
            (inventory['sku_id'] == sku_id)
        ]
        to_stock = to_inv['quantity_available'].iloc[0] if len(to_inv) > 0 else 0
        
        # Get forecast
        to_forecast = forecasts[
            (forecasts['warehouse_id'] == to_wh) & 
            (forecasts['sku_id'] == sku_id)
        ]
        forecast_demand = to_forecast['forecast'].sum() if len(to_forecast) > 0 else 0
        
        # Get SKU info
        sku_info = self.skus.loc[sku_id]
        safety_stock_days = sku_info.get('safety_stock_days', 7)
        safety_stock = forecast_demand * safety_stock_days / 14  # Approximate
        
        # Generate explanation
        explanation = (
            f"Transfer {qty} units of {sku_id} from {from_wh} to {to_wh}. "
            f"Reason: {to_wh} has current stock of {to_stock:.0f} units, "
            f"forecasted demand of {forecast_demand:.0f} units over the planning horizon, "
            f"and safety stock target of {safety_stock:.0f} units. "
            f"Current stock ({to_stock:.0f}) is below safety stock threshold, "
            f"requiring replenishment to prevent stockout risk."
        )
        
        if pd.notna(sku_info.get('perishability_ttl_days')):
            explanation += (
                f" Note: This SKU is perishable with TTL of {sku_info['perishability_ttl_days']} days."
            )
        
        return explanation
    
    def get_top_drivers(self, plan_row: Dict, inventory: pd.DataFrame,
                       forecasts: pd.DataFrame) -> List[str]:
        """Identify top 3 drivers of the decision."""
        to_wh = plan_row['to_wh']
        sku_id = plan_row['sku_id']
        
        # Get inventory and forecast
        to_inv = inventory[
            (inventory['warehouse_id'] == to_wh) & 
            (inventory['sku_id'] == sku_id)
        ]
        to_stock = to_inv['quantity_available'].iloc[0] if len(to_inv) > 0 else 0
        
        to_forecast = forecasts[
            (forecasts['warehouse_id'] == to_wh) & 
            (forecasts['sku_id'] == sku_id)
        ]
        forecast_demand = to_forecast['forecast'].sum() if len(to_forecast) > 0 else 0
        
        sku_info = self.skus.loc[sku_id]
        safety_stock_days = sku_info.get('safety_stock_days', 7)
        safety_stock = forecast_demand * safety_stock_days / 14
        
        drivers = []
        
        # Shortage risk
        if to_stock < safety_stock:
            shortage_risk = (safety_stock - to_stock) / max(safety_stock, 1) * 100
            drivers.append(f"Shortage risk: {shortage_risk:.1f}% below safety stock")
        
        # Transfer cost benefit
        drivers.append("Transfer cost benefit: Prevents expensive stockout penalties")
        
        # Perishability priority
        if pd.notna(sku_info.get('perishability_ttl_days')):
            drivers.append(f"Perishability priority: TTL {sku_info['perishability_ttl_days']} days")
        
        return drivers[:3]
    
    def calculate_delta_cost(self, plan_row: Dict, cost_model) -> float:
        """Estimate cost delta if transfer is not executed."""
        # Simplified: estimate stockout cost vs transfer cost
        to_wh = plan_row['to_wh']
        sku_id = plan_row['sku_id']
        qty = plan_row['qty']
        
        # Stockout cost if not transferred
        stockout_cost = cost_model.stockout_cost(to_wh, sku_id, qty)
        
        # Transfer cost
        transfer_cost = plan_row.get('estimated_cost', 0)
        
        # Delta = cost saved by transferring
        return stockout_cost - transfer_cost

