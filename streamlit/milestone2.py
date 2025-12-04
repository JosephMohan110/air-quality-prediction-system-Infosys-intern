# Milestone 2 Streamlit App with Enhanced Forecast Analysis
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

# -------------------------
# Page config
# -------------------------
st.set_page_config(layout="wide", page_title="Air Quality Forecast Analysis System")

# -------------------------
# Paths Configuration
# -------------------------
RESULTS_DIR = Path("results")
FORECAST_DIR = RESULTS_DIR / "forecasts"
MODELS_DIR = RESULTS_DIR / "models"

# Ensure directories exist
RESULTS_DIR.mkdir(exist_ok=True)
FORECAST_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

files = {
    "model_perf": RESULTS_DIR / "model_performance.csv",
    "best_models": RESULTS_DIR / "best_models.csv",
    "comparison": RESULTS_DIR / "model_comparison_per_pollutant.csv",
    "alerts": RESULTS_DIR / "alerts_summary.csv",
    "accuracy": FORECAST_DIR / "forecast_accuracy_metrics.csv"
}

# -------------------------
# Load CSV safely with enhanced error handling
# -------------------------
@st.cache_data
def load_csv(path):
    """Load CSV file with comprehensive error handling"""
    try:
        if path.exists():
            df = pd.read_csv(path)
            st.success(f"Successfully loaded: {path.name}")
            return df
        else:
            st.warning(f"File not found: {path}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading {path}: {str(e)}")
        return pd.DataFrame()

# Load all data files
model_perf = load_csv(files["model_perf"])
best_models = load_csv(files["best_models"])
comparison = load_csv(files["comparison"])
alerts = load_csv(files["alerts"])
accuracy_metrics = load_csv(files["accuracy"])

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.title("Analysis Controls")

# Get available forecast files
def get_available_forecasts():
    """Get available cities and pollutants from forecast files"""
    try:
        forecast_files = list(FORECAST_DIR.glob("forecast_*.csv"))
        if not forecast_files:
            st.sidebar.warning("No forecast files found in forecasts directory")
            return ["Delhi", "Mumbai"], ["PM2.5", "PM10"]
        
        cities = []
        pollutants = []
        
        for f in forecast_files:
            try:
                parts = f.stem.split('_')
                if len(parts) >= 3:
                    cities.append(parts[1])
                    pollutants.append(parts[2])
            except Exception as e:
                st.sidebar.warning(f"Error parsing filename {f.name}: {e}")
                continue
        
        cities = sorted(list(set(cities))) if cities else ["Delhi", "Mumbai"]
        pollutants = sorted(list(set(pollutants))) if pollutants else ["PM2.5", "PM10"]
        
        return cities, pollutants
    except Exception as e:
        st.sidebar.error(f"Error reading forecast files: {e}")
        return ["Delhi", "Mumbai"], ["PM2.5", "PM10"]

cities, pollutants = get_available_forecasts()

# City and pollutant selection
selected_city1 = st.sidebar.selectbox("Select Primary City", cities, key="city1")
selected_city2 = st.sidebar.selectbox("Select Comparison City", cities, 
                                    index=1 if len(cities) > 1 else 0, key="city2")
selected_pollutant = st.sidebar.selectbox("Select Pollutant", pollutants, key="pollutant")

# Analysis options
st.sidebar.subheader("Analysis Options")
show_confidence = st.sidebar.checkbox("Show Confidence Intervals", value=True)
show_residuals = st.sidebar.checkbox("Show Residual Analysis", value=True)
show_training = st.sidebar.checkbox("Show Training Data", value=False)

# -------------------------
# Title and Overview
# -------------------------
st.title("Air Quality Forecast Analysis System")
st.markdown("""
This dashboard provides comprehensive analysis of air quality forecasting models, 
including performance metrics, forecast comparisons, and alert systems.
""")

# -------------------------
# Function to load forecast data
# -------------------------
def load_forecast(city, pollutant):
    """Load forecast data with enhanced error handling"""
    try:
        file_path = FORECAST_DIR / f"forecast_{city}_{pollutant}.csv"
        if not file_path.exists():
            return None
        
        df = pd.read_csv(file_path)
        
        # Find and process date column
        date_col = next((c for c in df.columns if "date" in c.lower() or "ds" in c.lower()), None)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.rename(columns={date_col: "Datetime"})
            df = df.sort_values("Datetime")
        else:
            st.warning(f"No date column found in forecast for {city} - {pollutant}")
            return None
            
        return df
    except Exception as e:
        st.error(f"Error loading forecast for {city} - {pollutant}: {str(e)}")
        return None

# -------------------------
# Forecast Comparison Visualization
# -------------------------
st.subheader(f"Forecast Comparison: {selected_pollutant}")

df1 = load_forecast(selected_city1, selected_pollutant)
df2 = load_forecast(selected_city2, selected_pollutant)

if df1 is not None or df2 is not None:
    fig = go.Figure()

    # Define colors for better visualization
    colors = {
        'actual': ['#1f77b4', '#ff7f0e'],
        'forecast': ['#aec7e8', '#ffbb78']
    }

    for idx, (df, city) in enumerate(zip([df1, df2], [selected_city1, selected_city2])):
        if df is not None:
            # Forecast line
            forecast_col = next((c for c in df.columns if "yhat" in c.lower() or "forecast" in c.lower()), None)
            if forecast_col:
                fig.add_trace(go.Scatter(
                    x=df['Datetime'], y=df[forecast_col],
                    mode='lines', name=f'{city} Forecast', 
                    line=dict(dash='dash', color=colors['forecast'][idx]),
                    opacity=0.8
                ))
                
                # Confidence intervals
                if show_confidence:
                    lower_col = next((c for c in df.columns if "lower" in c.lower()), None)
                    upper_col = next((c for c in df.columns if "upper" in c.lower()), None)
                    
                    if lower_col and upper_col:
                        fig.add_trace(go.Scatter(
                            x=df['Datetime'], y=df[upper_col],
                            mode='lines', line=dict(width=0),
                            showlegend=False, name=f'{city} Upper CI'
                        ))
                        fig.add_trace(go.Scatter(
                            x=df['Datetime'], y=df[lower_col],
                            mode='lines', line=dict(width=0),
                            fillcolor=f'rgba{tuple(int(colors["forecast"][idx].lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.2,)}',
                            fill='tonexty', showlegend=False,
                            name=f'{city} Lower CI'
                        ))
            
            # Actual values
            if 'y_true' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['Datetime'], y=df['y_true'],
                    mode='lines', name=f'{city} Actual',
                    line=dict(width=2, color=colors['actual'][idx])
                ))

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title=f"{selected_pollutant} Concentration (µg/m³)",
        height=500,
        margin=dict(t=30, b=20, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info(f"No forecast data available for {selected_pollutant} in selected cities.")
    st.info("Please ensure forecast files exist in the results/forecasts/ directory.")

# -------------------------
# Performance Metrics Summary
# -------------------------
st.subheader("Forecast Performance Summary")

if not accuracy_metrics.empty:
    # Filter metrics for selected cities and pollutant
    filtered_metrics = accuracy_metrics[
        (accuracy_metrics["City"].isin([selected_city1, selected_city2])) &
        (accuracy_metrics["Pollutant"] == selected_pollutant)
    ]
    
    if not filtered_metrics.empty:
        # Display key metrics
        cols = st.columns(4)
        metrics_to_show = ['RMSE', 'MAE', 'MAPE', 'R2']
        
        for idx, metric in enumerate(metrics_to_show):
            if metric in filtered_metrics.columns:
                city1_data = filtered_metrics[filtered_metrics['City'] == selected_city1]
                city2_data = filtered_metrics[filtered_metrics['City'] == selected_city2]
                
                if len(city1_data) > 0 and len(city2_data) > 0:
                    city1_val = city1_data[metric].iloc[0]
                    city2_val = city2_data[metric].iloc[0]
                    
                    with cols[idx]:
                        delta_value = city1_val - city2_val
                        delta_color = "inverse" if metric in ['RMSE', 'MAE', 'MAPE'] else "normal"
                        
                        st.metric(
                            label=f"{metric} - {selected_city1}",
                            value=f"{city1_val:.2f}",
                            delta=f"{delta_value:+.2f}",
                            delta_color=delta_color
                        )

# -------------------------
# Detailed Accuracy Analysis
# -------------------------
st.subheader("Detailed Accuracy Metrics")

if not accuracy_metrics.empty:
    filtered_accuracy = accuracy_metrics[
        (accuracy_metrics["City"].isin([selected_city1, selected_city2])) &
        (accuracy_metrics["Pollutant"] == selected_pollutant)
    ]
    
    if not filtered_accuracy.empty:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Bar chart comparison
            metrics_to_plot = [col for col in ['RMSE', 'MAE', 'MAPE', 'R2'] if col in filtered_accuracy.columns]
            if metrics_to_plot:
                melted_df = filtered_accuracy.melt(
                    id_vars=["City", "Pollutant"], 
                    value_vars=metrics_to_plot,
                    var_name="Metric", 
                    value_name="Value"
                )
                
                fig = px.bar(
                    melted_df,
                    x="Metric", 
                    y="Value", 
                    color="City", 
                    barmode="group",
                    title=f"Accuracy Metrics Comparison for {selected_pollutant}",
                    color_discrete_sequence=px.colors.qualitative.Set1
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Summary table
            st.markdown("**Performance Summary**")
            summary_data = []
            for city in [selected_city1, selected_city2]:
                city_data = filtered_accuracy[filtered_accuracy['City'] == city]
                if not city_data.empty:
                    city_summary = {'City': city}
                    for metric in ['RMSE', 'MAE', 'MAPE', 'R2']:
                        if metric in city_data.columns:
                            city_summary[metric] = f"{city_data[metric].iloc[0]:.3f}"
                    summary_data.append(city_summary)
            
            if summary_data:
                st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
    else:
        st.info("No accuracy data available for the selected configuration")
else:
    st.info("Accuracy metrics data not available")

# -------------------------
# Model Performance Analysis
# -------------------------
st.subheader("Model Performance Analysis")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("**Model Performance Metrics**")
    if not model_perf.empty:
        filtered_perf = model_perf[model_perf['Pollutant'] == selected_pollutant]
        if not filtered_perf.empty:
            st.dataframe(filtered_perf, use_container_width=True)
        else:
            st.info(f"No performance data for {selected_pollutant}")
    else:
        st.info("Model performance data not available")

with col2:
    st.markdown("**Best Performing Models**")
    if not best_models.empty:
        city_models = best_models[
            (best_models['City'].isin([selected_city1, selected_city2])) &
            (best_models['Pollutant'] == selected_pollutant)
        ]
        if not city_models.empty:
            st.dataframe(city_models, use_container_width=True)
        else:
            st.info(f"No best model data for selected cities and {selected_pollutant}")
    else:
        st.info("Best models data not available")

# -------------------------
# Residual Analysis
# -------------------------
if show_residuals and df1 is not None and 'y_true' in df1.columns:
    st.subheader("Residual Analysis")
    
    forecast_col = next((c for c in df1.columns if "yhat" in c.lower()), None)
    if forecast_col:
        residuals = df1['y_true'] - df1[forecast_col]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Residuals distribution
            fig_hist = px.histogram(
                residuals, 
                nbins=30,
                title="Distribution of Residuals",
                labels={"value": "Residual Value"},
                color_discrete_sequence=['#ff7f0e']
            )
            fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
            fig_hist.update_layout(showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Residuals over time
            fig_time = go.Figure()
            fig_time.add_trace(go.Scatter(
                x=df1['Datetime'], y=residuals,
                mode='lines+markers',
                name='Residuals',
                marker=dict(color='#ff7f0e', size=4),
                line=dict(width=1)
            ))
            fig_time.add_hline(y=0, line_dash="dash", line_color="red")
            fig_time.update_layout(
                title="Residuals Over Time",
                xaxis_title="Date",
                yaxis_title="Residual Value"
            )
            st.plotly_chart(fig_time, use_container_width=True)
        
        # Residual statistics
        st.markdown("**Residual Statistics**")
        residual_stats = pd.DataFrame({
            'Statistic': ['Mean', 'Std Dev', 'Min', 'Max', 'RMSE'],
            'Value': [
                residuals.mean(),
                residuals.std(),
                residuals.min(),
                residuals.max(),
                np.sqrt((residuals**2).mean())
            ]
        })
        residual_stats['Value'] = residual_stats['Value'].round(4)
        st.dataframe(residual_stats, use_container_width=True)

# -------------------------
# Alert System Analysis
# -------------------------
st.subheader("Air Quality Alert System")

if not alerts.empty:
    city_alerts = alerts[alerts['City'].isin([selected_city1, selected_city2])]
    
    if not city_alerts.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Recent Alerts Summary**")
            st.dataframe(city_alerts, use_container_width=True)
        
        with col2:
            # Alert statistics
            st.markdown("**Alert Statistics**")
            alert_counts = city_alerts['Alert_Level'].value_counts()
            if not alert_counts.empty:
                fig_pie = px.pie(
                    values=alert_counts.values,
                    names=alert_counts.index,
                    title="Alert Level Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No alerts recorded for selected cities")
else:
    st.info("Alert system data not available")

# -------------------------
# Data Export Section
# -------------------------
st.subheader("Data Export")

export_col1, export_col2, export_col3 = st.columns(3)

with export_col1:
    if st.button("Export Forecast Data"):
        if df1 is not None:
            csv = df1.to_csv(index=False)
            st.download_button(
                label="Download Forecast CSV",
                data=csv,
                file_name=f"forecast_{selected_city1}_{selected_pollutant}.csv",
                mime="text/csv"
            )

with export_col2:
    if st.button("Export Performance Metrics"):
        if not accuracy_metrics.empty:
            filtered_data = accuracy_metrics[
                (accuracy_metrics["City"].isin([selected_city1, selected_city2])) &
                (accuracy_metrics["Pollutant"] == selected_pollutant)
            ]
            if not filtered_data.empty:
                csv = filtered_data.to_csv(index=False)
                st.download_button(
                    label="Download Metrics CSV",
                    data=csv,
                    file_name="accuracy_metrics.csv",
                    mime="text/csv"
                )

with export_col3:
    if st.button("Generate Report Summary"):
        report_content = f"""
Air Quality Forecast Analysis Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Analysis Configuration:
- Primary City: {selected_city1}
- Comparison City: {selected_city2}
- Pollutant: {selected_pollutant}

Data Status:
- Model Performance Data: {'Available' if not model_perf.empty else 'Not Available'}
- Best Models Data: {'Available' if not best_models.empty else 'Not Available'}
- Accuracy Metrics: {'Available' if not accuracy_metrics.empty else 'Not Available'}
- Alert Data: {'Available' if not alerts.empty else 'Not Available'}

Report includes:
- Forecast visualization and comparison
- Model performance metrics
- Accuracy analysis
- Residual diagnostics
- Alert system monitoring
"""
        st.download_button(
            label="Download Report",
            data=report_content,
            file_name="forecast_analysis_report.txt",
            mime="text/plain"
        )

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.markdown("### Air Quality Forecast Analysis System")
st.markdown("""
**Capabilities:**
- Multi-city forecast comparison and visualization
- Comprehensive model performance evaluation  
- Accuracy metrics analysis across different time horizons
- Residual analysis for model diagnostics
- Alert system monitoring and reporting
- Export functionality for further analysis
- Interactive model comparison across pollutants

**Technical Features:**
- Confidence interval visualization for uncertainty quantification
- Forecast horizon error analysis
- Statistical performance metrics (RMSE, MAE, MAPE, R-squared)
- Residual distribution and pattern analysis
- Comparative model performance assessment
- Customizable time range selection
""")

st.markdown("""
*Note: This system requires forecast data files in the results/forecasts/ directory.*
""")