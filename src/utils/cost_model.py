"""
Cost model utilities for transport, holding, stockout, and perishability costs.
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np


class CostModel:
    """Calculate various costs for replenishment planning."""
    
    def __init__(self, warehouses: pd.DataFrame, skus: pd.DataFrame, fleet: pd.DataFrame):
        """Initialize cost model with reference data."""
        self.warehouses = warehouses.set_index('warehouse_id')
        self.skus = skus.set_index('sku_id')
        self.fleet = fleet.set_index('truck_id')
        
    def calculate_distance(self, from_wh: str, to_wh: str) -> float:
        """Calculate distance between warehouses in km."""
        from_wh_data = self.warehouses.loc[from_wh]
        to_wh_data = self.warehouses.loc[to_wh]
        
        lat1, lon1 = from_wh_data['latitude'], from_wh_data['longitude']
        lat2, lon2 = to_wh_data['latitude'], to_wh_data['longitude']
        
        # Haversine formula
        R = 6371  # Earth radius in km
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = (np.sin(dlat/2)**2 + 
             np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c
    
    def transport_cost(self, from_wh: str, to_wh: str, quantity: int, 
                      sku_id: str, truck_type: str = 'Medium', 
                      expedited: bool = False) -> float:
        """Calculate transport cost for a transfer."""
        distance = self.calculate_distance(from_wh, to_wh)
        
        # Get truck cost per km
        truck_costs = {
            'Small': 1.0,
            'Medium': 1.5,
            'Large': 2.0,
            'Refrigerated': 2.5,
        }
        cost_per_km = truck_costs.get(truck_type, 1.5)
        
        # Expedited multiplier
        if expedited:
            cost_per_km *= 2.0
        
        # Base transport cost
        transport_cost = distance * cost_per_km
        
        # Fixed cost per trip
        fixed_cost = 100.0
        
        # Volume/weight based cost (simplified)
        sku_data = self.skus.loc[sku_id]
        volume_cost = quantity * sku_data['volume_m3_per_unit'] * 0.1
        weight_cost = quantity * sku_data['weight_kg_per_unit'] * 0.001
        
        return transport_cost + fixed_cost + volume_cost + weight_cost
    
    def holding_cost(self, warehouse_id: str, sku_id: str, quantity: int, days: int) -> float:
        """Calculate holding cost for inventory."""
        wh_data = self.warehouses.loc[warehouse_id]
        sku_data = self.skus.loc[sku_id]
        
        cost_per_unit_per_day = wh_data['holding_cost_per_unit_per_day']
        unit_cost = sku_data['unit_cost']
        
        # Holding cost = (inventory value) * holding_rate * days
        inventory_value = quantity * unit_cost
        holding_rate = cost_per_unit_per_day / unit_cost  # Convert to rate
        
        return inventory_value * holding_rate * days
    
    def stockout_cost(self, warehouse_id: str, sku_id: str, quantity: int) -> float:
        """Calculate stockout penalty cost."""
        wh_data = self.warehouses.loc[warehouse_id]
        sku_data = self.skus.loc[sku_id]
        
        stockout_cost_per_unit = wh_data['stockout_cost_per_unit']
        return quantity * stockout_cost_per_unit
    
    def perishability_waste_cost(self, sku_id: str, quantity: int, 
                                 days_in_transit: int) -> float:
        """Calculate waste cost if perishable items expire."""
        sku_data = self.skus.loc[sku_id]
        
        if pd.isna(sku_data['perishability_ttl_days']):
            return 0.0
        
        ttl_days = sku_data['perishability_ttl_days']
        
        if days_in_transit > ttl_days:
            # All items expire
            return quantity * sku_data['unit_cost']
        elif days_in_transit > ttl_days * 0.8:
            # Partial waste
            waste_ratio = (days_in_transit - ttl_days * 0.8) / (ttl_days * 0.2)
            return quantity * waste_ratio * sku_data['unit_cost']
        
        return 0.0
    
    def ordering_cost(self, quantity: int, sku_id: str) -> float:
        """Calculate ordering cost from supplier."""
        # Fixed ordering cost + variable cost
        fixed_cost = 50.0
        variable_cost = quantity * 0.1
        return fixed_cost + variable_cost
    
    def total_transfer_cost(self, from_wh: str, to_wh: str, sku_id: str, 
                           quantity: int, truck_type: str, days_in_transit: int,
                           expedited: bool = False) -> Dict[str, float]:
        """Calculate total cost breakdown for a transfer."""
        transport = self.transport_cost(from_wh, to_wh, quantity, sku_id, truck_type, expedited)
        waste = self.perishability_waste_cost(sku_id, quantity, days_in_transit)
        
        return {
            'transport_cost': transport,
            'perishability_waste_cost': waste,
            'total_cost': transport + waste,
        }

