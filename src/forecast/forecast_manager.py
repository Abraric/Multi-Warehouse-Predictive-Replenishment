"""
Forecasting module for SKU×warehouse demand prediction.

Supports Prophet (time-series) and LightGBM (ML) models with
prediction intervals and feature engineering.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
import pickle
import json

try:
    from prophet import Prophet
except ImportError:
    Prophet = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

from sklearn.metrics import mean_absolute_error, mean_squared_error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ForecastManager:
    """Manages demand forecasting models per SKU×warehouse."""
    
    def __init__(self, data_dir: Path, out_dir: Path):
        """Initialize forecast manager."""
        self.data_dir = Path(data_dir)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.models = {}
        self.metrics = {}
        
    def load_data(self) -> pd.DataFrame:
        """Load processed sales data."""
        filepath = self.data_dir / 'daily_sales_aggregated.csv'
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        return df
    
    def prepare_prophet_data(self, series: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for Prophet model."""
        prophet_df = series[['date', 'quantity_sold']].copy()
        prophet_df.columns = ['ds', 'y']
        prophet_df = prophet_df.sort_values('ds')
        return prophet_df
    
    def train_prophet_model(self, series: pd.DataFrame) -> Optional[Prophet]:
        """Train Prophet model on time series."""
        if Prophet is None:
            logger.warning("Prophet not available, skipping Prophet model")
            return None
        
        try:
            prophet_df = self.prepare_prophet_data(series)
            
            if len(prophet_df) < 14:
                logger.warning(f"Insufficient data for Prophet: {len(prophet_df)} rows")
                return None
            
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                interval_width=0.95,
            )
            model.fit(prophet_df)
            return model
        except Exception as e:
            logger.error(f"Error training Prophet model: {e}")
            return None
    
    def prepare_lightgbm_features(self, series: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for LightGBM model."""
        features = [
            'rolling_7d_mean', 'rolling_7d_std', 'rolling_28d_mean', 'rolling_28d_std',
            'lag_1d', 'lag_7d', 'lag_28d',
            'day_of_week', 'day_of_month', 'month',
            'is_weekend', 'is_holiday_season', 'is_summer', 'is_promotion'
        ]
        
        available_features = [f for f in features if f in series.columns]
        
        X = series[available_features].fillna(0)
        y = series['quantity_sold']
        
        return X, y
    
    def train_lightgbm_model(self, series: pd.DataFrame) -> Optional[lgb.Booster]:
        """Train LightGBM model on features."""
        if lgb is None:
            logger.warning("LightGBM not available, skipping LightGBM model")
            return None
        
        try:
            X, y = self.prepare_lightgbm_features(series)
            
            if len(X) < 30:
                logger.warning(f"Insufficient data for LightGBM: {len(X)} rows")
                return None
            
            # Split train/test
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            train_data = lgb.Dataset(X_train, label=y_train)
            
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'verbose': -1,
            }
            
            model = lgb.train(
                params,
                train_data,
                num_boost_round=100,
                valid_sets=[train_data],
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
            )
            
            # Evaluate
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            logger.info(f"LightGBM metrics - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
            
            return model
        except Exception as e:
            logger.error(f"Error training LightGBM model: {e}")
            return None
    
    def train_models(self):
        """Train models for all SKU×warehouse combinations."""
        logger.info("Loading data for training...")
        df = self.load_data()
        
        logger.info(f"Training models for {len(df.groupby(['warehouse_id', 'sku_id']))} SKU×warehouse combinations...")
        
        for (warehouse_id, sku_id), group in df.groupby(['warehouse_id', 'sku_id']):
            key = f"{warehouse_id}_{sku_id}"
            group = group.sort_values('date')
            
            # Train Prophet
            prophet_model = self.train_prophet_model(group)
            
            # Train LightGBM
            lightgbm_model = self.train_lightgbm_model(group)
            
            self.models[key] = {
                'warehouse_id': warehouse_id,
                'sku_id': sku_id,
                'prophet': prophet_model,
                'lightgbm': lightgbm_model,
                'last_training_date': group['date'].max(),
            }
        
        logger.info(f"Training complete. Trained {len(self.models)} models")
    
    def predict_prophet(self, model: Prophet, start_date: datetime, horizon: int) -> pd.DataFrame:
        """Generate Prophet forecast."""
        future = model.make_future_dataframe(periods=horizon)
        forecast = model.predict(future)
        
        # Filter to horizon
        forecast = forecast[forecast['ds'] >= start_date].head(horizon)
        
        return pd.DataFrame({
            'date': forecast['ds'],
            'forecast': forecast['yhat'].values,
            'lower': forecast['yhat_lower'].values,
            'upper': forecast['yhat_upper'].values,
        })
    
    def predict_lightgbm(self, model: lgb.Booster, series: pd.DataFrame, 
                        start_date: datetime, horizon: int) -> pd.DataFrame:
        """Generate LightGBM forecast."""
        # Use last available features
        last_row = series.iloc[-1:].copy()
        
        predictions = []
        current_features = last_row.copy()
        
        for i in range(horizon):
            # Prepare features
            X, _ = self.prepare_lightgbm_features(current_features)
            
            # Predict
            pred = model.predict(X)[0]
            pred = max(0, pred)  # Non-negative
            
            # Update features for next prediction (simplified)
            new_row = current_features.iloc[-1:].copy()
            new_row['lag_1d'] = pred
            new_row['date'] = start_date + timedelta(days=i)
            new_row['day_of_week'] = new_row['date'].dt.dayofweek.values[0]
            new_row['day_of_month'] = new_row['date'].dt.day.values[0]
            new_row['month'] = new_row['date'].dt.month.values[0]
            new_row['is_weekend'] = (new_row['day_of_week'] >= 5).astype(int).values[0]
            
            predictions.append({
                'date': start_date + timedelta(days=i),
                'forecast': pred,
                'lower': pred * 0.8,  # Simplified intervals
                'upper': pred * 1.2,
            })
            
            current_features = pd.concat([current_features, new_row], ignore_index=True)
        
        return pd.DataFrame(predictions)
    
    def predict(self, warehouse_id: str, sku_id: str, start_date: datetime, 
                horizon: int, model_type: str = 'prophet') -> pd.DataFrame:
        """Generate forecast for specific SKU×warehouse."""
        key = f"{warehouse_id}_{sku_id}"
        
        if key not in self.models:
            logger.warning(f"No model found for {key}")
            return pd.DataFrame()
        
        model_info = self.models[key]
        
        if model_type == 'prophet' and model_info['prophet'] is not None:
            return self.predict_prophet(model_info['prophet'], start_date, horizon)
        elif model_type == 'lightgbm' and model_info['lightgbm'] is not None:
            # Need historical data for features
            df = self.load_data()
            series = df[(df['warehouse_id'] == warehouse_id) & 
                       (df['sku_id'] == sku_id)].sort_values('date')
            return self.predict_lightgbm(model_info['lightgbm'], series, start_date, horizon)
        else:
            logger.warning(f"Model type {model_type} not available for {key}")
            return pd.DataFrame()
    
    def generate_all_forecasts(self, start_date: datetime, horizon: int):
        """Generate forecasts for all SKU×warehouse combinations."""
        logger.info(f"Generating forecasts from {start_date} for {horizon} days...")
        
        all_forecasts = []
        
        for key, model_info in self.models.items():
            warehouse_id = model_info['warehouse_id']
            sku_id = model_info['sku_id']
            
            # Try Prophet first, fallback to LightGBM
            forecast = self.predict(warehouse_id, sku_id, start_date, horizon, 'prophet')
            if forecast.empty:
                forecast = self.predict(warehouse_id, sku_id, start_date, horizon, 'lightgbm')
            
            if not forecast.empty:
                forecast['warehouse_id'] = warehouse_id
                forecast['sku_id'] = sku_id
                forecast['model_type'] = 'prophet' if not forecast.empty else 'lightgbm'
                all_forecasts.append(forecast)
        
        if all_forecasts:
            result = pd.concat(all_forecasts, ignore_index=True)
            result.to_csv(self.out_dir / 'forecasts.csv', index=False)
            logger.info(f"Saved forecasts: {len(result)} rows")
        else:
            logger.warning("No forecasts generated")
    
    def save_models(self):
        """Save trained models to disk."""
        models_dir = self.out_dir / 'models'
        models_dir.mkdir(exist_ok=True)
        
        for key, model_info in self.models.items():
            model_file = models_dir / f"{key}.pkl"
            # Save metadata only (models are complex to pickle)
            metadata = {
                'warehouse_id': model_info['warehouse_id'],
                'sku_id': model_info['sku_id'],
                'last_training_date': str(model_info['last_training_date']),
            }
            with open(model_file, 'w') as f:
                json.dump(metadata, f)
        
        logger.info(f"Saved model metadata to {models_dir}")


def main():
    parser = argparse.ArgumentParser(description='Forecast demand for SKU×warehouse')
    parser.add_argument('--train', action='store_true',
                      help='Train models')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                      help='Directory with processed data')
    parser.add_argument('--out-dir', type=str, default='data/forecasts',
                      help='Output directory for forecasts')
    parser.add_argument('--horizon', type=int, default=14,
                      help='Forecast horizon in days')
    
    args = parser.parse_args()
    
    manager = ForecastManager(args.data_dir, args.out_dir)
    
    if args.train:
        manager.train_models()
        manager.save_models()
    
    # Generate forecasts
    start_date = datetime.now()
    manager.generate_all_forecasts(start_date, args.horizon)


if __name__ == '__main__':
    main()

