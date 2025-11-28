"""
Script to generate output directory with sample data, reports, and visuals.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.synthetic_generator import SyntheticDataGenerator
from src.etl.etl import ETLPipeline
from src.forecast.forecast_manager import ForecastManager
from src.optimization.replenishment_optimizer import ReplenishmentOptimizer
from src.simulator.transfer_simulator import TransferSimulator


def generate_output():
    """Generate all output files."""
    print("Generating output directory contents...")
    
    # Create directories
    output_dir = Path("output")
    sample_data_dir = output_dir / "sample_data"
    reports_dir = output_dir / "reports"
    visuals_dir = output_dir / "visuals"
    
    for dir_path in [sample_data_dir, reports_dir, visuals_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Generate sample data
    print("1. Generating sample data...")
    generator = SyntheticDataGenerator(seed=42)
    start_date = datetime.now() - timedelta(days=180)
    data = generator.generate_all(
        n_warehouses=5,
        n_skus=50,
        days=180,
        start_date=start_date
    )
    
    # Save sample CSVs
    sample_data_dir.mkdir(parents=True, exist_ok=True)
    data['warehouses'].to_csv(sample_data_dir / "warehouses.csv", index=False)
    data['skus'].to_csv(sample_data_dir / "skus.csv", index=False)
    # Sample of sales history
    sample_sales = data['sales_history'].head(1000)
    sample_sales.to_csv(sample_data_dir / "sales_history_sample.csv", index=False)
    
    print(f"   Saved warehouses.csv ({len(data['warehouses'])} rows)")
    print(f"   Saved skus.csv ({len(data['skus'])} rows)")
    print(f"   Saved sales_history_sample.csv ({len(sample_sales)} rows)")
    
    # Step 2: Run ETL
    print("2. Running ETL...")
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    generator.save_to_csv(data, raw_dir)
    
    processed_dir = Path("data/processed")
    pipeline = ETLPipeline()
    pipeline.process(raw_dir, processed_dir)
    
    # Step 3: Generate forecasts
    print("3. Generating forecasts...")
    forecasts_dir = Path("data/forecasts")
    manager = ForecastManager(processed_dir, forecasts_dir)
    try:
        manager.train_models()
        manager.save_models()
        manager.generate_all_forecasts(datetime.now(), horizon=14)
    except Exception as e:
        print(f"   Warning: Forecasting had issues: {e}")
        # Create dummy forecasts for demo
        forecasts = []
        for _, wh in data['warehouses'].iterrows():
            for _, sku in data['skus'].iterrows():
                for i in range(14):
                    forecasts.append({
                        'date': (datetime.now() + timedelta(days=i)).date(),
                        'warehouse_id': wh['warehouse_id'],
                        'sku_id': sku['sku_id'],
                        'forecast': np.random.uniform(10, 50),
                        'lower': np.random.uniform(8, 40),
                        'upper': np.random.uniform(15, 60),
                    })
        forecasts_df = pd.DataFrame(forecasts)
        forecasts_dir.mkdir(parents=True, exist_ok=True)
        forecasts_df.to_csv(forecasts_dir / 'forecasts.csv', index=False)
    
    # Step 4: Run optimization
    print("4. Running optimization...")
    warehouses = pd.read_csv(processed_dir / 'warehouses.csv')
    skus = pd.read_csv(processed_dir / 'skus.csv')
    fleet = data['fleet']
    
    optimizer = ReplenishmentOptimizer(
        processed_dir / 'inventory_snapshot.csv',
        forecasts_dir / 'forecasts.csv',
        warehouses,
        skus,
        fleet
    )
    
    plan = optimizer.optimize(horizon_days=14)
    plan.to_csv(reports_dir / 'replenishment_plan_sample.csv', index=False)
    print(f"   Generated plan with {len(plan)} transfers")
    
    # Step 5: Run simulation
    print("5. Running simulation...")
    simulator = TransferSimulator(
        reports_dir / 'replenishment_plan_sample.csv',
        warehouses,
        skus,
        fleet,
        seed=42
    )
    
    inventory = pd.read_csv(processed_dir / 'inventory_snapshot.csv')
    forecasts = pd.read_csv(forecasts_dir / 'forecasts.csv')
    sim_report = simulator.generate_report(inventory, forecasts)
    # Save transfers DataFrame
    if 'transfers' in sim_report and len(sim_report['transfers']) > 0:
        sim_report['transfers'].to_csv(reports_dir / 'simulation_report_sample.csv', index=False)
        print(f"   Simulation complete: service level = {sim_report['summary']['overall_service_level']:.2%}")
    else:
        # Create empty report if no transfers
        pd.DataFrame().to_csv(reports_dir / 'simulation_report_sample.csv', index=False)
        print("   Simulation complete (no transfers to simulate)")
    
    # Step 6: Generate visuals
    print("6. Generating visuals...")
    
    # Forecast example
    if len(forecasts) > 0:
        sample_forecast = forecasts[
            (forecasts['warehouse_id'] == forecasts['warehouse_id'].iloc[0]) &
            (forecasts['sku_id'] == forecasts['sku_id'].iloc[0])
        ].copy()
        sample_forecast['date'] = pd.to_datetime(sample_forecast['date'])
        
        plt.figure(figsize=(12, 6))
        plt.plot(sample_forecast['date'], sample_forecast['forecast'], label='Forecast', linewidth=2)
        plt.fill_between(sample_forecast['date'], sample_forecast['lower'], sample_forecast['upper'],
                         alpha=0.3, label='Prediction Interval')
        plt.xlabel('Date')
        plt.ylabel('Forecasted Demand')
        plt.title(f"Demand Forecast: {sample_forecast['sku_id'].iloc[0]} at {sample_forecast['warehouse_id'].iloc[0]}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(visuals_dir / 'forecast_example.png', dpi=150)
        plt.close()
        print("   Saved forecast_example.png")
    
    # Plan Gantt
    if len(plan) > 0:
        plan['ship_date'] = pd.to_datetime(plan['ship_date'])
        plan['expected_arrival'] = pd.to_datetime(plan['expected_arrival'])
        
        fig, ax = plt.subplots(figsize=(14, max(6, len(plan) * 0.3)))
        for idx, row in plan.iterrows():
            y_pos = idx
            duration = (row['expected_arrival'] - row['ship_date']).days
            ax.barh(y_pos, duration, left=row['ship_date'], alpha=0.7)
        
        ax.set_yticks(range(len(plan)))
        ax.set_yticklabels([f"{r['from_wh']} → {r['to_wh']} ({r['sku_id']})" for _, r in plan.iterrows()])
        ax.set_xlabel('Date')
        ax.set_title('Replenishment Plan Timeline (Gantt)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(visuals_dir / 'plan_gantt.png', dpi=150)
        plt.close()
        print("   Saved plan_gantt.png")
        
        # Cost breakdown
        cost_by_destination = plan.groupby('to_wh')['estimated_cost'].sum().sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        cost_by_destination.plot(kind='bar')
        plt.xlabel('Destination Warehouse')
        plt.ylabel('Total Cost ($)')
        plt.title('Cost Breakdown by Destination Warehouse')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(visuals_dir / 'cost_breakdown.png', dpi=150)
        plt.close()
        print("   Saved cost_breakdown.png")
    else:
        # Create placeholder visuals if no plan
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No transfers in plan', ha='center', va='center', fontsize=16)
        ax.set_title('Replenishment Plan Timeline')
        plt.savefig(visuals_dir / 'plan_gantt.png', dpi=150)
        plt.close()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No cost data', ha='center', va='center', fontsize=16)
        ax.set_title('Cost Breakdown')
        plt.savefig(visuals_dir / 'cost_breakdown.png', dpi=150)
        plt.close()
    
    print("\nOutput generation complete!")
    print(f"Check {output_dir} for all generated files.")


if __name__ == '__main__':
    generate_output()

