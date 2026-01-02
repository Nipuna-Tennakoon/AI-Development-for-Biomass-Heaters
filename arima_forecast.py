"""
ARIMA Forecasting Script
Forecasts "Average T.F. Heater Serpentine Coil Inlet Temperature" using ARIMA method.
Saves plots and metrics (MSE, RMSE, MAE) in ARIMA folder.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
from datetime import date
from statsmodels.tsa.arima.model import ARIMA

# Configuration
DATA_PATH = 'dataset/final_df_002.csv'
TARGET_COLUMN = 'Average T.F. Heater Serpentine Coil Inlet Temperature'
ARIMA_ORDER = (2, 1, 4)  # (p, d, q) - AR order, differencing order, MA order
TRAIN_SIZE = 0.8  # 80% for training, 20% for testing
OUTPUT_FOLDER = 'ARIMA'

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("=" * 60)
print("ARIMA Forecasting")
print("=" * 60)

# Load data
print("\n1. Loading data...")
df = pd.read_csv(DATA_PATH)
df['Date Time'] = pd.to_datetime(df['Date Time'])
print(f"   Data loaded: {len(df)} rows, {len(df.columns)} columns")
print(f"   Date range: {df['Date Time'].min()} to {df['Date Time'].max()}")

# Check if target column exists
if TARGET_COLUMN not in df.columns:
    raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset. Available columns: {df.columns.tolist()}")

# Extract target variable
y = df[TARGET_COLUMN].copy()
dates = df['Date Time'].copy()

# Split data into train and test sets
print("\n2. Splitting data into train and test sets...")
train_size = int(len(df) * TRAIN_SIZE)
train_df = df.iloc[:train_size].copy()
test_df = df.iloc[train_size:].copy()

y_train = train_df[TARGET_COLUMN].copy()
y_test = test_df[TARGET_COLUMN].copy()
dates_train = train_df['Date Time'].copy()
dates_test = test_df['Date Time'].copy()

print(f"   Training set: {len(y_train)} samples ({TRAIN_SIZE*100:.0f}%)")
print(f"   Test set: {len(y_test)} samples ({(1-TRAIN_SIZE)*100:.0f}%)")

# ARIMA Forecasting
print("\n3. Performing ARIMA forecasting...")
print(f"   ARIMA order (p, d, q): {ARIMA_ORDER}")

# Fit ARIMA model on training data
print("   Fitting ARIMA model on training data...")
try:
    arima_model = ARIMA(y_train, order=ARIMA_ORDER)
    arima_model_fit = arima_model.fit()
    print(f"   Model fitted successfully")
    print(f"   Model summary: AIC = {arima_model_fit.aic:.2f}, BIC = {arima_model_fit.bic:.2f}")
except Exception as e:
    print(f"   Error fitting ARIMA model: {e}")
    raise

# Calculate training predictions (in-sample)
print("   Calculating training predictions...")
y_train_pred = arima_model_fit.fittedvalues
# ARIMA fittedvalues might have different index due to differencing
# Align with full training set
y_train_pred_aligned = pd.Series(index=y_train.index, dtype=float)

# The fitted values start after differencing (d parameter)
start_idx = ARIMA_ORDER[1]  # d parameter
if len(y_train_pred) > 0:
    # Get fitted values as array
    fitted_vals = y_train_pred.values if hasattr(y_train_pred, 'values') else np.array(y_train_pred)
    # Assign to appropriate positions
    end_idx = min(start_idx + len(fitted_vals), len(y_train))
    if end_idx > start_idx:
        y_train_pred_aligned.iloc[start_idx:end_idx] = fitted_vals[:end_idx-start_idx]
    # Fill the first values with the first available prediction
    if start_idx > 0 and len(fitted_vals) > 0:
        y_train_pred_aligned.iloc[:start_idx] = fitted_vals[0]
    # Forward/backward fill any remaining NaN values
    y_train_pred_aligned = y_train_pred_aligned.ffill().bfill()
else:
    # Fallback: use mean of training data
    y_train_pred_aligned = pd.Series(y_train.mean(), index=y_train.index)

y_train_pred = y_train_pred_aligned

# Forecast on test set
print("   Forecasting on test set...")
y_pred = arima_model_fit.forecast(steps=len(y_test))
y_pred = pd.Series(y_pred, index=y_test.index)

# Calculate metrics
print("\n4. Calculating metrics...")

# Training metrics
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = math.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)

# Test metrics
test_mse = mean_squared_error(y_test, y_pred)
test_rmse = math.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_pred)

print("   Training Metrics:")
print(f"     MSE:  {train_mse:.4f}")
print(f"     RMSE: {train_rmse:.4f}")
print(f"     MAE:  {train_mae:.4f}")
print("   Test Metrics:")
print(f"     MSE:  {test_mse:.4f}")
print(f"     RMSE: {test_rmse:.4f}")
print(f"     MAE:  {test_mae:.4f}")

# Save metrics to file
metrics_file = os.path.join(OUTPUT_FOLDER, 'metrics.txt')
with open(metrics_file, 'w') as f:
    f.write("ARIMA Forecasting Metrics\n")
    f.write("=" * 40 + "\n\n")
    f.write(f"Target Variable: {TARGET_COLUMN}\n")
    f.write(f"ARIMA Order (p, d, q): {ARIMA_ORDER}\n")
    f.write(f"Train Size: {len(y_train)} samples ({TRAIN_SIZE*100:.0f}%)\n")
    f.write(f"Test Size: {len(y_test)} samples ({(1-TRAIN_SIZE)*100:.0f}%)\n")
    f.write(f"Model AIC: {arima_model_fit.aic:.4f}\n")
    f.write(f"Model BIC: {arima_model_fit.bic:.4f}\n\n")
    f.write("Training Performance Metrics:\n")
    f.write(f"  MSE:  {train_mse:.6f}\n")
    f.write(f"  RMSE: {train_rmse:.6f}\n")
    f.write(f"  MAE:  {train_mae:.6f}\n\n")
    f.write("Test Performance Metrics:\n")
    f.write(f"  MSE:  {test_mse:.6f}\n")
    f.write(f"  RMSE: {test_rmse:.6f}\n")
    f.write(f"  MAE:  {test_mae:.6f}\n")

print(f"\n   Metrics saved to: {metrics_file}")

# Create visualizations
print("\n5. Creating visualizations...")

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# Plot 1: Full time series with forecast
fig, ax = plt.subplots(figsize=(16, 8))

# Plot training data (last portion for visibility)
train_plot_size = min(500, len(y_train))
sns.lineplot(x=dates_train.iloc[-train_plot_size:], 
             y=y_train.iloc[-train_plot_size:], 
             ax=ax, color='dodgerblue', label='Training Data', linewidth=1.5, alpha=0.7)

# Plot test ground truth
sns.lineplot(x=dates_test, y=y_test, 
             ax=ax, color='gold', label='Actual (Test)', linewidth=2)

# Plot predictions
sns.lineplot(x=dates_test, y=y_pred, 
             ax=ax, color='darkorange', label='Forecasted (ARIMA)', linewidth=2)

ax.set_title(f'ARIMA Forecasting: {TARGET_COLUMN}\nOrder (p,d,q) = {ARIMA_ORDER} | Test MSE: {test_mse:.2f}, RMSE: {test_rmse:.2f}, MAE: {test_mae:.2f}', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('Date Time', fontsize=12)
ax.set_ylabel('Temperature', fontsize=12)
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot1_path = os.path.join(OUTPUT_FOLDER, 'forecast_full.png')
plt.savefig(plot1_path, bbox_inches='tight')
print(f"   Full forecast plot saved to: {plot1_path}")
plt.close()

# Plot 2: Zoomed view of test period
fig, ax = plt.subplots(figsize=(16, 8))

sns.lineplot(x=dates_test, y=y_test, 
             ax=ax, color='gold', label='Actual (Test)', linewidth=2.5, marker='o', markersize=4)
sns.lineplot(x=dates_test, y=y_pred, 
             ax=ax, color='darkorange', label='Forecasted (ARIMA)', linewidth=2.5, marker='s', markersize=4)

ax.set_title(f'ARIMA Forecasting - Test Period Zoom\nOrder (p,d,q) = {ARIMA_ORDER} | Test MSE: {test_mse:.2f}, RMSE: {test_rmse:.2f}, MAE: {test_mae:.2f}', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('Date Time', fontsize=12)
ax.set_ylabel('Temperature', fontsize=12)
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot2_path = os.path.join(OUTPUT_FOLDER, 'forecast_test_zoom.png')
plt.savefig(plot2_path, bbox_inches='tight')
print(f"   Test period zoom plot saved to: {plot2_path}")
plt.close()

# Plot 3: Residuals plot
fig, ax = plt.subplots(figsize=(16, 6))

residuals = y_test - y_pred
sns.lineplot(x=dates_test, y=residuals, ax=ax, color='indianred', linewidth=1.5)
ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax.set_title(f'Residuals Plot (Actual - Forecasted)\nMean Residual: {residuals.mean():.4f}, Std Residual: {residuals.std():.4f}', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('Date Time', fontsize=12)
ax.set_ylabel('Residuals', fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot3_path = os.path.join(OUTPUT_FOLDER, 'residuals.png')
plt.savefig(plot3_path, bbox_inches='tight')
print(f"   Residuals plot saved to: {plot3_path}")
plt.close()

# Plot 4: Scatter plot - Actual vs Predicted
fig, ax = plt.subplots(figsize=(10, 10))

ax.scatter(y_test, y_pred, alpha=0.6, s=50, color='darkorange', edgecolors='black', linewidth=0.5)

# Perfect prediction line
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

ax.set_xlabel('Actual Values', fontsize=12)
ax.set_ylabel('Predicted Values', fontsize=12)
ax.set_title(f'Actual vs Predicted Values\nTest MSE: {test_mse:.2f}, RMSE: {test_rmse:.2f}, MAE: {test_mae:.2f}', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot4_path = os.path.join(OUTPUT_FOLDER, 'actual_vs_predicted.png')
plt.savefig(plot4_path, bbox_inches='tight')
print(f"   Actual vs Predicted scatter plot saved to: {plot4_path}")
plt.close()

# Save predictions to CSV
predictions_df = pd.DataFrame({
    'Date Time': dates_test,
    'Actual': y_test.values,
    'Forecasted': y_pred.values,
    'Residual': residuals.values
})
predictions_file = os.path.join(OUTPUT_FOLDER, 'predictions.csv')
predictions_df.to_csv(predictions_file, index=False)
print(f"   Predictions saved to: {predictions_file}")

print("\n" + "=" * 60)
print("Forecasting completed successfully!")
print("=" * 60)
print(f"\nAll outputs saved in '{OUTPUT_FOLDER}' folder:")
print(f"  - metrics.txt")
print(f"  - predictions.csv")
print(f"  - forecast_full.png")
print(f"  - forecast_test_zoom.png")
print(f"  - residuals.png")
print(f"  - actual_vs_predicted.png")
print("\n")

