"""
Streamlit dashboard for multi-warehouse replenishment system.

Provides interactive views for warehouse KPIs, forecasts, plans, and simulations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Multi-Warehouse Replenishment Dashboard",
    page_icon="📦",
    layout="wide"
)

DATA_DIR = Path("data/processed")
FORECASTS_DIR = Path("data/forecasts")
PLANS_DIR = Path("output/reports")


@st.cache_data
def load_data():
    """Load reference data."""
    try:
        warehouses = pd.read_csv(DATA_DIR / "warehouses.csv")
        skus = pd.read_csv(DATA_DIR / "skus.csv")
        inventory = pd.read_csv(DATA_DIR / "inventory_snapshot.csv")
        forecasts = pd.read_csv(FORECASTS_DIR / "forecasts.csv") if (FORECASTS_DIR / "forecasts.csv").exists() else pd.DataFrame()
        return warehouses, skus, inventory, forecasts
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None


def warehouse_kpis_page():
    """Warehouse KPIs and inventory levels."""
    st.header("📊 Warehouse KPIs")
    
    warehouses, skus, inventory, forecasts = load_data()
    
    if warehouses is None:
        st.error("Data not available. Please run data generation and ETL first.")
        return
    
    # Overall metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_warehouses = len(warehouses)
    total_skus = len(skus)
    total_inventory_value = (inventory['quantity_available'] * 
                            inventory.merge(skus[['sku_id', 'unit_cost']], on='sku_id')['unit_cost']).sum()
    avg_capacity_usage = (inventory.merge(skus[['sku_id', 'volume_m3_per_unit']], on='sku_id')
                          .groupby('warehouse_id')
                          .apply(lambda x: (x['quantity_available'] * x['volume_m3_per_unit']).sum())
                          .mean() / warehouses['capacity_volume_m3'].mean() * 100)
    
    col1.metric("Warehouses", total_warehouses)
    col2.metric("SKUs", total_skus)
    col3.metric("Total Inventory Value", f"${total_inventory_value:,.0f}")
    col4.metric("Avg Capacity Usage", f"{avg_capacity_usage:.1f}%")
    
    # Warehouse map
    st.subheader("Warehouse Locations")
    fig = px.scatter_mapbox(
        warehouses,
        lat="latitude",
        lon="longitude",
        hover_name="name",
        hover_data=["city", "capacity_volume_m3"],
        zoom=3,
        height=400,
    )
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)
    
    # Inventory by warehouse
    st.subheader("Inventory Levels by Warehouse")
    inv_summary = inventory.groupby('warehouse_id').agg({
        'quantity_available': 'sum'
    }).reset_index()
    inv_summary = inv_summary.merge(warehouses[['warehouse_id', 'name']], on='warehouse_id')
    
    fig = px.bar(inv_summary, x='name', y='quantity_available', 
                 title="Total Inventory Units by Warehouse")
    st.plotly_chart(fig, use_container_width=True)
    
    # Capacity usage
    st.subheader("Capacity Usage")
    capacity_data = inventory.merge(skus[['sku_id', 'volume_m3_per_unit']], on='sku_id')
    capacity_usage = capacity_data.groupby('warehouse_id').apply(
        lambda x: (x['quantity_available'] * x['volume_m3_per_unit']).sum()
    ).reset_index()
    capacity_usage.columns = ['warehouse_id', 'used_volume']
    capacity_usage = capacity_usage.merge(warehouses[['warehouse_id', 'capacity_volume_m3', 'name']], on='warehouse_id')
    capacity_usage['usage_pct'] = (capacity_usage['used_volume'] / capacity_usage['capacity_volume_m3'] * 100)
    
    fig = px.bar(capacity_usage, x='name', y='usage_pct',
                 title="Capacity Usage % by Warehouse")
    st.plotly_chart(fig, use_container_width=True)


def forecast_explorer_page():
    """Interactive forecast explorer."""
    st.header("🔮 Forecast Explorer")
    
    warehouses, skus, inventory, forecasts = load_data()
    
    if warehouses is None or forecasts.empty:
        st.error("Forecasts not available. Please run forecasting first.")
        return
    
    # Filters
    col1, col2 = st.columns(2)
    selected_warehouse = col1.selectbox("Select Warehouse", warehouses['warehouse_id'].unique())
    selected_sku = col2.selectbox("Select SKU", skus['sku_id'].unique())
    
    # Filter forecasts
    filtered_forecasts = forecasts[
        (forecasts['warehouse_id'] == selected_warehouse) &
        (forecasts['sku_id'] == selected_sku)
    ].copy()
    
    if filtered_forecasts.empty:
        st.warning("No forecasts available for this combination.")
        return
    
    filtered_forecasts['date'] = pd.to_datetime(filtered_forecasts['date'])
    filtered_forecasts = filtered_forecasts.sort_values('date')
    
    # Plot forecast
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=filtered_forecasts['date'],
        y=filtered_forecasts['forecast'],
        name='Forecast',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=filtered_forecasts['date'],
        y=filtered_forecasts['lower'],
        name='Lower Bound',
        line=dict(color='gray', dash='dash'),
        fill=None
    ))
    
    fig.add_trace(go.Scatter(
        x=filtered_forecasts['date'],
        y=filtered_forecasts['upper'],
        name='Upper Bound',
        line=dict(color='gray', dash='dash'),
        fill='tonexty',
        fillcolor='rgba(128,128,128,0.2)'
    ))
    
    fig.update_layout(
        title=f"Demand Forecast: {selected_sku} at {selected_warehouse}",
        xaxis_title="Date",
        yaxis_title="Forecasted Demand",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Forecast summary
    st.subheader("Forecast Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Forecast", f"{filtered_forecasts['forecast'].sum():.0f}")
    col2.metric("Average Daily", f"{filtered_forecasts['forecast'].mean():.1f}")
    col3.metric("Max Daily", f"{filtered_forecasts['forecast'].max():.0f}")


def plan_viewer_page():
    """Replenishment plan viewer."""
    st.header("📋 Replenishment Plan Viewer")
    
    # List available plans
    plan_files = sorted(PLANS_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    
    if not plan_files:
        st.error("No plans found. Please run optimization first.")
        return
    
    selected_plan = st.selectbox("Select Plan", [p.stem for p in plan_files])
    plan_path = PLANS_DIR / f"{selected_plan}.csv"
    plan = pd.read_csv(plan_path)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transfers", len(plan))
    col2.metric("Total Cost", f"${plan['estimated_cost'].sum():,.2f}")
    col3.metric("Avg Cost per Transfer", f"${plan['estimated_cost'].mean():,.2f}")
    col4.metric("Unique SKUs", plan['sku_id'].nunique())
    
    # Plan table
    st.subheader("Transfer Details")
    st.dataframe(plan, use_container_width=True)
    
    # Gantt chart
    st.subheader("Transfer Timeline (Gantt)")
    if len(plan) > 0:
        plan['ship_date'] = pd.to_datetime(plan['ship_date'])
        plan['expected_arrival'] = pd.to_datetime(plan['expected_arrival'])
        plan['duration'] = (plan['expected_arrival'] - plan['ship_date']).dt.days
        
        fig = go.Figure()
        
        for idx, row in plan.iterrows():
            fig.add_trace(go.Bar(
                x=[row['duration']],
                y=[f"{row['from_wh']} → {row['to_wh']}"],
                base=row['ship_date'],
                orientation='h',
                name=f"{row['sku_id']}",
                text=f"Qty: {row['qty']}",
                textposition='inside'
            ))
        
        fig.update_layout(
            title="Transfer Timeline",
            xaxis_title="Date",
            yaxis_title="Transfer",
            barmode='overlay',
            height=max(400, len(plan) * 30)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Cost breakdown
        st.subheader("Cost Breakdown")
        cost_by_wh = plan.groupby('to_wh')['estimated_cost'].sum().reset_index()
        fig = px.pie(cost_by_wh, values='estimated_cost', names='to_wh',
                    title="Cost Distribution by Destination Warehouse")
        st.plotly_chart(fig, use_container_width=True)


def simulation_viewer_page():
    """Simulation results viewer."""
    st.header("🎲 Simulation Results")
    
    plan_files = sorted(PLANS_DIR.glob("*_simulation.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    
    if not plan_files:
        st.error("No simulation results found. Please run simulation first.")
        return
    
    selected_sim = st.selectbox("Select Simulation", [p.stem for p in plan_files])
    sim_path = PLANS_DIR / f"{selected_sim}.csv"
    sim_results = pd.read_csv(sim_path)
    
    # Filter summary rows
    summary = sim_results[sim_results['type'] == 'SUMMARY']
    transfers = sim_results[sim_results['type'] == 'TRANSFER']
    stockouts = sim_results[sim_results['type'] == 'STOCKOUT']
    
    # Display summary
    st.subheader("Simulation Summary")
    for _, row in summary.iterrows():
        st.metric(row['metric'], f"{row['value']:.2f}", help=row['details'])
    
    # Transfer execution details
    if len(transfers) > 0:
        st.subheader("Transfer Execution Details")
        st.dataframe(transfers, use_container_width=True)
    
    # Stockout events
    if len(stockouts) > 0:
        st.subheader("Stockout Events")
        st.dataframe(stockouts, use_container_width=True)
        
        # Stockout chart
        fig = px.bar(stockouts, x='warehouse_id', y='value',
                    title="Stockout Quantities by Warehouse")
        st.plotly_chart(fig, use_container_width=True)


def main():
    """Main dashboard app."""
    st.title("📦 Multi-Warehouse Predictive Replenishment Dashboard")
    
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Navigation",
        ["Warehouse KPIs", "Forecast Explorer", "Plan Viewer", "Simulation Results"]
    )
    
    if page == "Warehouse KPIs":
        warehouse_kpis_page()
    elif page == "Forecast Explorer":
        forecast_explorer_page()
    elif page == "Plan Viewer":
        plan_viewer_page()
    elif page == "Simulation Results":
        simulation_viewer_page()


if __name__ == "__main__":
    main()

