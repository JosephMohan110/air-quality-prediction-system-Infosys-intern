# streamlit/milestone3.py
"""
AirAware Excellence - Premium Air Quality Forecasting Platform
Enhanced with automatic forecasts and user input features
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.colors import sequential, qualitative
import pickle
import warnings
from datetime import datetime, timedelta
import json

warnings.filterwarnings('ignore')

# ----------------------------
# Configuration
# ----------------------------
ROOT = Path.cwd()
MODELS_DIR = ROOT / "models"

# Professional Color Scheme
COLORS = {
    'primary': '#1e3a8a',
    'primary_light': '#3b82f6',
    'primary_dark': '#1e40af',
    'secondary': '#dc2626',
    'success': '#059669',
    'warning': '#d97706',
    'danger': '#dc2626',
    'dark': '#0f172a',
    'dark_light': '#334155',
    'light': '#f8fafc',
    'accent': '#7c3aed',
    'background': '#f1f5f9'
}

AQI_COLORS = {
    'Good': '#059669',
    'Satisfactory': '#0d9488',
    'Moderate': '#ca8a04',
    'Poor': '#ea580c',
    'Very Poor': '#dc2626',
    'Severe': '#7f1d1d'
}

AQI_RANGES = {
    'Good': (0, 50),
    'Satisfactory': (51, 100),
    'Moderate': (101, 200),
    'Poor': (201, 300),
    'Very Poor': (301, 400),
    'Severe': (401, 500)
}

# Clean CSS Styling
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {{
        font-family: 'Inter', sans-serif;
    }}
    
    .main-header {{
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['accent']} 100%);
        color: white;
        padding: 2.5rem 2rem;
        margin: -1rem -1rem 2rem -1rem;
        text-align: center;
        border-radius: 0 0 20px 20px;
    }}
    
    .main-title {{
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: white;
    }}
    
    .main-subtitle {{
        font-size: 1.2rem;
        font-weight: 300;
        opacity: 0.9;
        margin-top: 0.5rem;
        color: #e2e8f0;
    }}
    
    .section-header {{
        font-size: 1.5rem;
        font-weight: 600;
        color: {COLORS['dark']};
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid {COLORS['primary_light']};
        border-radius: 0 0 8px 8px;
    }}
    
    .metric-card {{
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin: 0.5rem 0;
        transition: transform 0.2s ease;
    }}
    
    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.12);
    }}
    
    .prediction-card {{
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['primary_dark']} 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }}
    
    .city-selector {{
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }}
    
    .forecast-container {{
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }}
    
    .user-input-container {{
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #bae6fd;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.1);
    }}
    
    .aqi-meter {{
        background: linear-gradient(90deg, 
            {AQI_COLORS['Good']} 0% 10%, 
            {AQI_COLORS['Satisfactory']} 10% 20%, 
            {AQI_COLORS['Moderate']} 20% 40%, 
            {AQI_COLORS['Poor']} 40% 60%, 
            {AQI_COLORS['Very Poor']} 60% 80%, 
            {AQI_COLORS['Severe']} 80% 100%);
        height: 12px;
        border-radius: 6px;
        margin: 1rem 0;
        position: relative;
    }}
    
    .aqi-indicator {{
        position: absolute;
        top: -8px;
        width: 4px;
        height: 28px;
        background: {COLORS['dark']};
        border-radius: 2px;
        transform: translateX(-50%);
    }}
    
    .day-card {{
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 1rem 0.5rem;
        border-radius: 10px;
        border: 2px solid #e2e8f0;
        text-align: center;
        transition: all 0.3s ease;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        flex: 1;
    }}
    
    .day-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.12);
        border-color: {COLORS['primary_light']};
    }}
    
    .day-card-today {{
        border-color: {COLORS['primary']};
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border-width: 3px;
    }}
    
    .day-name {{
        font-size: 0.85rem;
        font-weight: 600;
        color: {COLORS['dark']};
        margin-bottom: 0.2rem;
        line-height: 1.2;
    }}
    
    .day-date {{
        font-size: 0.75rem;
        color: {COLORS['dark_light']};
        margin-bottom: 0.5rem;
        line-height: 1.2;
    }}
    
    .day-aqi {{
        font-size: 1.3rem;
        font-weight: 700;
        margin: 0.3rem 0;
        line-height: 1.2;
    }}
    
    .day-category {{
        font-size: 0.7rem;
        font-weight: 500;
        padding: 0.2rem 0.4rem;
        border-radius: 12px;
        display: inline-block;
        line-height: 1.2;
        max-width: 90px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# XGBoost Model Loading
# ----------------------------
@st.cache_resource
def load_xgboost_model():
    """Load the XGBoost model from the models directory"""
    try:
        model_path = MODELS_DIR / "xgboost_models" / "xgboost_model.pkl"
        
        if not model_path.exists():
            st.error(f"Model file not found at: {model_path}")
            return None
            
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            
        st.success("XGBoost model loaded successfully!")
        return model_data
        
    except Exception as e:
        st.error(f"Error loading XGBoost model: {str(e)}")
        return None

model_data = load_xgboost_model()

# ----------------------------
# Core Functions
# ----------------------------
def get_available_cities():
    """Get list of cities available in the model"""
    if model_data is None:
        return []
    daily_cities = list(model_data.get('daily_models', {}).keys())
    return sorted(list(set(daily_cities)))

def get_city_pollutants(city, frequency='daily'):
    """Get available pollutants for a city"""
    if model_data is None:
        return []
    
    models_key = f'{frequency}_models'
    if city in model_data.get(models_key, {}):
        return list(model_data[models_key][city].keys())
    return []

def create_prediction_features(city, periods, frequency='daily', user_inputs=None):
    """Create feature matrix for prediction with optional user inputs"""
    features = []
    current_date = datetime.now()
    
    for i in range(periods):
        if frequency == 'daily':
            date = current_date + timedelta(days=i)
        else:
            date = current_date + timedelta(hours=i)
            
        # Base features
        feature_row = {
            'year': date.year, 'month': date.month, 'day': date.day,
            'dayofweek': date.weekday(), 'quarter': (date.month - 1) // 3 + 1,
            'dayofyear': date.timetuple().tm_yday, 'weekofyear': date.isocalendar()[1],
            'month_sin': np.sin(2 * np.pi * date.month / 12),
            'month_cos': np.cos(2 * np.pi * date.month / 12),
            'day_sin': np.sin(2 * np.pi * date.day / 31),
            'day_cos': np.cos(2 * np.pi * date.day / 31),
            'dayofweek_sin': np.sin(2 * np.pi * date.weekday() / 7),
            'dayofweek_cos': np.cos(2 * np.pi * date.weekday() / 7),
        }
        
        # Add pollutant values (use user inputs if provided, else defaults)
        pollutant_defaults = {
            'pm10': 80.0, 'no': 10.0, 'no2': 30.0, 'nox': 40.0, 'nh3': 15.0,
            'co': 1.0, 'so2': 15.0, 'o3': 40.0, 'benzene': 2.0, 'toluene': 5.0,
            'xylene': 3.0, 'aqi': 120.0, 'pm2_5': 60.0
        }
        
        for poll, default_val in pollutant_defaults.items():
            if user_inputs and poll in user_inputs:
                feature_row[poll] = user_inputs[poll]
            else:
                feature_row[poll] = default_val
        
        if frequency == 'hourly':
            feature_row.update({
                'hour': date.hour,
                'hour_sin': np.sin(2 * np.pi * date.hour / 24),
                'hour_cos': np.cos(2 * np.pi * date.hour / 24)
            })
            
        features.append(feature_row)
    
    return pd.DataFrame(features)

def predict_pollutant(city, pollutant, periods=7, frequency='daily', user_inputs=None):
    """Make prediction for a specific pollutant"""
    if (model_data is None or city not in model_data.get(f'{frequency}_models', {}) or 
        pollutant not in model_data[f'{frequency}_models'][city]):
        return None
    
    try:
        model_info = model_data[f'{frequency}_models'][city][pollutant]
        model = model_info['model']
        feature_scaler = model_info['feature_scaler']
        target_scaler = model_info['target_scaler']
        feature_columns = model_info['feature_columns']
        
        features_df = create_prediction_features(city, periods, frequency, user_inputs)
        
        # Ensure all required columns are present
        for col in feature_columns:
            if col not in features_df.columns:
                features_df[col] = 0.0
        
        X_pred = features_df[feature_columns]
        X_scaled = feature_scaler.transform(X_pred)
        
        y_pred_scaled = model.predict(X_scaled)
        y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        if frequency == 'daily':
            dates = [datetime.now() + timedelta(days=i) for i in range(periods)]
        else:
            dates = [datetime.now() + timedelta(hours=i) for i in range(periods)]
            
        return pd.DataFrame({
            'Date': dates, 'Pollutant': pollutant, 'Predicted_Value': y_pred, 'Frequency': frequency
        })
        
    except Exception as e:
        st.error(f"Prediction error for {pollutant}: {str(e)}")
        return None

def calculate_aqi_from_pollutants(pollutant_values):
    """Calculate AQI from individual pollutant concentrations"""
    if not pollutant_values:
        return 0
        
    aqi_components = []
    for poll, value in pollutant_values.items():
        if poll == 'pm2_5': aqi_component = min(value * 2, 500)
        elif poll == 'pm10': aqi_component = min(value * 1.5, 500)
        elif poll in ['no2', 'so2']: aqi_component = min(value * 2.5, 500)
        elif poll == 'o3': aqi_component = min(value * 3, 500)
        elif poll == 'co': aqi_component = min(value * 50, 500)
        else: aqi_component = min(value * 2, 500)
        aqi_components.append(aqi_component)
    
    return np.mean(aqi_components) if aqi_components else 0

def get_aqi_category(aqi_value):
    """Get AQI category and color based on AQI value"""
    for category, (low, high) in AQI_RANGES.items():
        if low <= aqi_value <= high:
            return category, AQI_COLORS[category]
    return "Unknown", "#666666"

def generate_city_forecast(city, frequency='daily', periods=7, user_inputs=None):
    """Generate automatic forecast for a city with optional user inputs"""
    if model_data is None:
        return None, None
        
    pollutants = get_city_pollutants(city, frequency)
    if not pollutants:
        return None, None
    
    all_predictions = {}
    for pollutant in pollutants:
        pred_df = predict_pollutant(city, pollutant, periods, frequency, user_inputs)
        if pred_df is not None and not pred_df.empty:
            all_predictions[pollutant] = pred_df
    
    current_pollutants = {}
    for poll, df in all_predictions.items():
        if not df.empty and 'Predicted_Value' in df.columns:
            current_pollutants[poll] = df.iloc[0]['Predicted_Value']
    
    if not current_pollutants:
        return None, None
    
    # Calculate AQI for each day
    daily_aqi_predictions = []
    for i in range(periods):
        day_pollutants = {}
        for poll, df in all_predictions.items():
            if not df.empty and i < len(df):
                day_pollutants[poll] = df.iloc[i]['Predicted_Value']
        
        if day_pollutants:
            day_aqi = calculate_aqi_from_pollutants(day_pollutants)
            daily_aqi_predictions.append(day_aqi)
    
    current_aqi = calculate_aqi_from_pollutants(current_pollutants)
    category, color = get_aqi_category(current_aqi)
    
    return all_predictions, {
        'aqi': current_aqi, 
        'category': category, 
        'color': color,
        'pollutant_count': len(all_predictions),
        'current_pollutants': current_pollutants,
        'daily_aqi_predictions': daily_aqi_predictions
    }

# ----------------------------
# Enhanced Visualization Functions
# ----------------------------
def create_aqi_gauge_with_meter(aqi_value, category, color, city, frequency):
    """Create enhanced AQI gauge chart with meter line"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=aqi_value,
        number={'font': {'size': 42, 'color': color, 'family': "Inter"}, 'suffix': " AQI"},
        delta={'reference': 100, 'increasing': {'color': COLORS['danger']}, 'decreasing': {'color': COLORS['success']}},
        gauge={
            'axis': {'range': [0, 500], 'tickwidth': 2, 'tickcolor': COLORS['dark'], 'tickfont': {'size': 12}},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 3,
            'bordercolor': COLORS['dark_light'],
            'steps': [
                {'range': [0, 50], 'color': AQI_COLORS['Good']},
                {'range': [51, 100], 'color': AQI_COLORS['Satisfactory']},
                {'range': [101, 200], 'color': AQI_COLORS['Moderate']},
                {'range': [201, 300], 'color': AQI_COLORS['Poor']},
                {'range': [301, 400], 'color': AQI_COLORS['Very Poor']},
                {'range': [401, 500], 'color': AQI_COLORS['Severe']}
            ],
            'threshold': {
                'line': {'color': COLORS['dark'], 'width': 4},
                'thickness': 0.75,
                'value': aqi_value
            }
        },
        title={'text': f"{city} - {category}", 'font': {'size': 20, 'family': "Inter", 'color': COLORS['dark']}}
    ))
    
    fig.update_layout(
        height=350, 
        margin=dict(l=50, r=50, t=100, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': "Inter"}
    )
    return fig

def create_aqi_meter_line(aqi_value):
    """Create a horizontal AQI meter line with indicator"""
    # Calculate position percentage (0-100%)
    position = min(aqi_value / 5.0, 100)  # AQI max is 500, so divide by 5
    
    st.markdown(f"""
    <div style="margin: 1rem 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem; font-size: 0.8rem; color: {COLORS['dark_light']};">
            <span>0</span>
            <span>50</span>
            <span>100</span>
            <span>200</span>
            <span>300</span>
            <span>400</span>
            <span>500</span>
        </div>
        <div class="aqi-meter">
            <div class="aqi-indicator" style="left: {position}%;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 0.5rem; font-size: 0.7rem; color: {COLORS['dark_light']};">
            <span style="color: {AQI_COLORS['Good']}">Good</span>
            <span style="color: {AQI_COLORS['Satisfactory']}">Satisfactory</span>
            <span style="color: {AQI_COLORS['Moderate']}">Moderate</span>
            <span style="color: {AQI_COLORS['Poor']}">Poor</span>
            <span style="color: {AQI_COLORS['Very Poor']}">Very Poor</span>
            <span style="color: {AQI_COLORS['Severe']}">Severe</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_enhanced_forecast_timeline(predictions_dict, city, frequency, aqi_predictions=None):
    """Create enhanced forecast timeline with AQI predictions"""
    if not predictions_dict:
        fig = go.Figure()
        fig.update_layout(title="No data available", height=400)
        return fig
    
    # Create subplots
    if aqi_predictions:
        fig = sp.make_subplots(
            rows=2, cols=1,
            subplot_titles=('Pollutants Forecast', 'AQI Forecast Trend'),
            vertical_spacing=0.15,
            row_heights=[0.6, 0.4]
        )
    else:
        fig = sp.make_subplots(
            rows=1, cols=1,
            subplot_titles=('Pollutants Forecast',)
        )
    
    # Get dates from first prediction
    first_pred = next(iter(predictions_dict.values()))
    dates = first_pred['Date'] if not first_pred.empty else []
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Add pollutants to first subplot
    for i, (pollutant, df) in enumerate(predictions_dict.items()):
        if not df.empty:
            fig.add_trace(
                go.Scatter(
                    x=df['Date'], y=df['Predicted_Value'],
                    mode='lines+markers', 
                    name=pollutant.upper(),
                    line=dict(color=colors[i % len(colors)], width=3),
                    marker=dict(size=6, symbol='circle'),
                    hovertemplate=f'{pollutant.upper()}<br>Date: %{{x}}<br>Value: %{{y:.1f}}<extra></extra>'
                ),
                row=1, col=1
            )
    
    # Add AQI predictions to second subplot if available
    if aqi_predictions and len(aqi_predictions) == len(dates):
        aqi_colors = [get_aqi_category(aqi)[1] for aqi in aqi_predictions]
        
        fig.add_trace(
            go.Scatter(
                x=dates, y=aqi_predictions,
                mode='lines+markers',
                name='Predicted AQI',
                line=dict(color=COLORS['primary'], width=4),
                marker=dict(size=8, color=aqi_colors),
                hovertemplate='Predicted AQI<br>Date: %{{x}}<br>AQI: %{{y:.0f}}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Add AQI range bands
        for category, (low, high) in AQI_RANGES.items():
            fig.add_hrect(
                y0=low, y1=high,
                fillcolor=AQI_COLORS[category],
                opacity=0.2,
                line_width=0,
                row=2, col=1
            )
    
    fig.update_layout(
        height=600 if aqi_predictions else 400,
        title_text=f"{city} - {frequency.title()} Air Quality Forecast",
        showlegend=True,
        hovermode='x unified',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': "Inter"}
    )
    
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Concentration", row=1, col=1)
    
    if aqi_predictions:
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="AQI", row=2, col=1)
    
    return fig

def create_7day_forecast_cards(aqi_predictions, dates):
    """Create attractive 7-day forecast cards using columns"""
    if not aqi_predictions or len(aqi_predictions) < 7:
        return
    
    day_names = ['Today', 'Tomorrow', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7']
    
    # Create 7 columns for the days
    cols = st.columns(7)
    
    for i, col in enumerate(cols):
        if i < len(aqi_predictions):
            aqi = aqi_predictions[i]
            category, color = get_aqi_category(aqi)
            date_str = dates[i].strftime('%b %d') if i < len(dates) else f"Day {i+1}"
            
            # Special styling for today
            today_class = "day-card-today" if i == 0 else ""
            
            with col:
                st.markdown(f"""
                <div class="day-card {today_class}">
                    <div class="day-name">{day_names[i]}</div>
                    <div class="day-date">{date_str}</div>
                    <div class="day-aqi" style="color: {color};">{aqi:.0f}</div>
                    <div class="day-category" style="background-color: {color}; color: white;">{category}</div>
                </div>
                """, unsafe_allow_html=True)

def create_forecast_table(predictions_dict, aqi_predictions=None, days=7):
    """Create enhanced forecast table with AQI trends"""
    if not predictions_dict:
        return None
    
    first_pred = next(iter(predictions_dict.values()))
    dates = first_pred['Date'] if not first_pred.empty else []
    
    forecast_data = []
    for i, date in enumerate(dates):
        if i >= days:
            break
            
        day_data = {'Date': date.strftime('%Y-%m-%d')}
        day_pollutants = {}
        
        for poll, df in predictions_dict.items():
            if not df.empty and i < len(df):
                value = df.iloc[i]['Predicted_Value']
                day_data[poll.upper()] = f"{value:.1f}"
                day_pollutants[poll] = value
        
        if day_pollutants:
            # Use calculated AQI if provided, otherwise calculate
            if aqi_predictions and i < len(aqi_predictions):
                day_aqi = aqi_predictions[i]
            else:
                day_aqi = calculate_aqi_from_pollutants(day_pollutants)
                
            day_category, day_color = get_aqi_category(day_aqi)
            day_data['AQI'] = f"{day_aqi:.0f}"
            day_data['Category'] = day_category
            day_data['_aqi_value'] = day_aqi
            day_data['_color'] = day_color
            forecast_data.append(day_data)
    
    if not forecast_data:
        return None
        
    df = pd.DataFrame(forecast_data)
    return df

# ----------------------------
# User Input Section
# ----------------------------
def create_user_input_section():
    """Create section for user to input pollutant values"""
    st.markdown("""
    <div class="user-input-container">
        <h2 style="color: #1e293b; margin-bottom: 1rem;">Custom Pollution Input</h2>
        <p style="color: #475569; margin-bottom: 1.5rem;">Adjust current pollutant levels to see how they affect AQI forecasts</p>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    user_inputs = {}
    
    with col1:
        user_inputs['pm2_5'] = st.slider("PM2.5 (µg/m³)", 0.0, 500.0, 60.0, help="Fine particulate matter")
        user_inputs['pm10'] = st.slider("PM10 (µg/m³)", 0.0, 500.0, 80.0, help="Coarse particulate matter")
        user_inputs['no2'] = st.slider("NO2 (µg/m³)", 0.0, 200.0, 30.0, help="Nitrogen dioxide")
    
    with col2:
        user_inputs['o3'] = st.slider("O3 (µg/m³)", 0.0, 200.0, 40.0, help="Ozone")
        user_inputs['co'] = st.slider("CO (mg/m³)", 0.0, 10.0, 1.0, help="Carbon monoxide")
        user_inputs['so2'] = st.slider("SO2 (µg/m³)", 0.0, 200.0, 15.0, help="Sulfur dioxide")
    
    with col3:
        user_inputs['nh3'] = st.slider("NH3 (µg/m³)", 0.0, 100.0, 15.0, help="Ammonia")
        user_inputs['benzene'] = st.slider("Benzene (µg/m³)", 0.0, 20.0, 2.0, help="Benzene")
        user_inputs['toluene'] = st.slider("Toluene (µg/m³)", 0.0, 50.0, 5.0, help="Toluene")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    return user_inputs

# ----------------------------
# Streamlit Application
# ----------------------------
st.set_page_config(
    page_title="AirAware - Premium Air Quality Forecasting",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Header
st.markdown("""
<div class="main-header">
    <div class="main-title">AirAware Excellence</div>
    <div class="main-subtitle">Advanced AI-Powered Air Quality Forecasting & Analytics</div>
    <div style="margin-top: 1rem; font-size: 1rem; opacity: 0.8;">
        Real-time predictions • Machine Learning • Environmental Intelligence
    </div>
</div>
""", unsafe_allow_html=True)

# Main Layout with Tabs
tab1, tab2 = st.tabs(["City Forecast", "Custom Analysis"])

with tab1:
    col1, col2 = st.columns([1, 3])

    with col1:
        # Control Panel
        st.markdown("""
        <div class="city-selector">
            <h3 style="color: #1e293b; margin-bottom: 1.5rem;">Control Panel</h3>
        """, unsafe_allow_html=True)
        
        # Model Status
        if model_data:
            st.success("Model Loaded Successfully")
            available_cities = get_available_cities()
            st.metric("Available Cities", len(available_cities))
        else:
            st.error("Model Not Available")
            st.info("Please ensure your model file is available at: models/xgboost_models/xgboost_model.pkl")
        
        # City Selection
        available_cities = get_available_cities()
        if available_cities:
            selected_city = st.selectbox(
                "Select City",
                available_cities,
                index=available_cities.index('Aizawl') if 'Aizawl' in available_cities else 0,
                help="Choose a city for air quality forecasting"
            )
            
            # Forecast Settings
            frequency = st.radio(
                "Forecast Frequency", 
                ["Daily", "Hourly"], 
                horizontal=True,
                help="Select forecast frequency"
            ).lower()
            
            if frequency == 'daily':
                periods = st.slider("Forecast Days", 1, 14, 7, help="Number of days to forecast")
            else:
                periods = st.slider("Forecast Hours", 1, 72, 24, help="Number of hours to forecast")
            
            # Quick Actions
            st.markdown("---")
            st.markdown("**Quick Actions**")
            if st.button("Refresh Forecast", width='stretch'):
                st.rerun()
                
        else:
            selected_city = None
            st.warning("No cities available in the model")
        
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        # Main Content Area
        if selected_city and model_data:
            # Generate automatic forecast
            predictions, aqi_info = generate_city_forecast(selected_city, frequency, periods)
            
            if predictions and aqi_info:
                # Current Status Section
                st.markdown("""
                <div class="forecast-container">
                    <h2 style="color: #1e293b; margin-bottom: 1.5rem;">Current Air Quality Status</h2>
                """, unsafe_allow_html=True)
                
                # Current AQI and Metrics in columns
                col_a, col_b, col_c = st.columns([2, 1, 1])
                
                with col_a:
                    gauge_fig = create_aqi_gauge_with_meter(
                        aqi_info['aqi'], aqi_info['category'], aqi_info['color'], selected_city, frequency
                    )
                    st.plotly_chart(gauge_fig, use_container_width=True)
                    
                    # Add AQI meter line below the gauge
                    st.markdown("<div style='margin-top: -2rem;'>", unsafe_allow_html=True)
                    create_aqi_meter_line(aqi_info['aqi'])
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col_b:
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h3 style="margin: 0 0 0.5rem 0; font-size: 1.1rem;">Current AQI</h3>
                        <h1 style="margin: 0; font-size: 2.8rem; color: {aqi_info['color']};">{aqi_info['aqi']:.0f}</h1>
                        <p style="margin: 0.5rem 0 0 0; font-weight: 600; color: {aqi_info['color']}; font-size: 1.1rem;">{aqi_info['category']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_c:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3 style="color: {COLORS['primary']}; margin: 0; font-size: 1.8rem;">{aqi_info['pollutant_count']}</h3>
                        <p style="color: {COLORS['dark_light']}; margin: 0.5rem 0 0 0;">Pollutants Tracked</p>
                    </div>
                    <div class="metric-card">
                        <h3 style="color: {COLORS['success']}; margin: 0; font-size: 1.8rem;">{periods}</h3>
                        <p style="color: {COLORS['dark_light']}; margin: 0.5rem 0 0 0;">Forecast Periods</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # 7-Day Forecast Cards
                if frequency == 'daily' and aqi_info.get('daily_aqi_predictions'):
                    st.markdown("""
                    <div class="forecast-container">
                        <h2 style="color: #1e293b; margin-bottom: 1.5rem;">7-Day AQI Forecast Overview</h2>
                    """, unsafe_allow_html=True)
                    
                    first_pred = next(iter(predictions.values()))
                    dates = first_pred['Date'] if not first_pred.empty else []
                    create_7day_forecast_cards(aqi_info['daily_aqi_predictions'], dates)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Forecast Visualization Section
                st.markdown("""
                <div class="forecast-container">
                    <h2 style="color: #1e293b; margin-bottom: 1.5rem;">Forecast Visualization</h2>
                """, unsafe_allow_html=True)
                
                timeline_fig = create_enhanced_forecast_timeline(
                    predictions, selected_city, frequency, aqi_info.get('daily_aqi_predictions')
                )
                st.plotly_chart(timeline_fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Detailed Forecast Section
                st.markdown("""
                <div class="forecast-container">
                    <h2 style="color: #1e293b; margin-bottom: 1.5rem;">Detailed Forecast Data</h2>
                """, unsafe_allow_html=True)
                
                forecast_table = create_forecast_table(
                    predictions, 
                    aqi_info.get('daily_aqi_predictions'), 
                    days=7
                )
                if forecast_table is not None:
                    # Style the dataframe
                    styled_df = forecast_table.drop(['_aqi_value', '_color'], axis=1, errors='ignore')
                    st.dataframe(styled_df, use_container_width=True, height=400)
                    
                    # Download button
                    csv = forecast_table.to_csv(index=False)
                    st.download_button(
                        label="Download Forecast Data",
                        data=csv,
                        file_name=f"{selected_city}_{frequency}_forecast.csv",
                        mime="text/csv",
                        width='stretch'
                    )
                else:
                    st.info("No forecast data available for the selected parameters")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
            else:
                st.warning(f"No forecast data available for {selected_city}")
                st.info("Try selecting a different city or check the model configuration")
        
        elif not model_data:
            st.error("""
            ## XGBoost Model Not Available
            
            Please ensure your model file is properly configured at:
            models/xgboost_models/xgboost_model.pkl
            
            The application requires the trained XGBoost model to generate forecasts.
            """)

with tab2:
    # Custom Analysis Tab
    st.markdown("""
    <div class="forecast-container">
        <h2 style="color: #1e293b; margin-bottom: 1rem;">Custom Pollution Analysis</h2>
        <p style="color: #475569; margin-bottom: 2rem;">
        Adjust current pollutant levels to simulate different scenarios and see their impact on AQI forecasts.
        </p>
    """, unsafe_allow_html=True)
    
    if model_data:
        # User input section
        user_inputs = create_user_input_section()
        
        # Generate forecast with user inputs
        if st.button("Generate Custom Forecast", width='stretch', type="primary"):
            with st.spinner("Generating custom forecast..."):
                # Calculate AQI from user inputs
                custom_aqi = calculate_aqi_from_pollutants(user_inputs)
                category, color = get_aqi_category(custom_aqi)
                
                st.success("Custom AQI calculated successfully!")
                
                # Show custom AQI results
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    gauge_fig = create_aqi_gauge_with_meter(
                        custom_aqi, category, color, "Custom Scenario", "Analysis"
                    )
                    st.plotly_chart(gauge_fig, use_container_width=True)
                    
                    # Add AQI meter line
                    create_aqi_meter_line(custom_aqi)
                
                with col2:
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h3 style="margin: 0 0 0.5rem 0; font-size: 1.1rem;">Custom AQI</h3>
                        <h1 style="margin: 0; font-size: 2.8rem; color: {color};">{custom_aqi:.0f}</h1>
                        <p style="margin: 0.5rem 0 0 0; font-weight: 600; color: {color}; font-size: 1.1rem;">{category}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show pollutant impact
                    st.markdown("""
                    <div class="metric-card">
                        <h4 style="color: #1e293b; margin: 0 0 1rem 0;">Pollutant Impact</h4>
                    """, unsafe_allow_html=True)
                    
                    # Show top contributors to AQI
                    pollutant_impact = []
                    for poll, value in user_inputs.items():
                        if poll == 'pm2_5': impact = value * 2
                        elif poll == 'pm10': impact = value * 1.5
                        elif poll in ['no2', 'so2']: impact = value * 2.5
                        elif poll == 'o3': impact = value * 3
                        elif poll == 'co': impact = value * 50
                        else: impact = value * 2
                        pollutant_impact.append((poll.upper(), impact))
                    
                    # Sort by impact
                    pollutant_impact.sort(key=lambda x: x[1], reverse=True)
                    
                    for poll, impact in pollutant_impact[:3]:
                        st.metric(f"{poll} Contribution", f"{impact:.1f}")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
    else:
        st.warning("Please load the model first to use custom analysis features")

# Enhanced Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 2rem;'>
    <p style='margin: 0; font-size: 1rem; font-weight: 600;'>AirAware Excellence - Environmental Intelligence Platform</p>
    <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem;'>
        Powered by XGBoost Machine Learning • Real-time Analytics • Predictive Insights
    </p>
    <p style='margin: 0.5rem 0 0 0; font-size: 0.8rem; opacity: 0.7;'>
        Making the world's air quality transparent and predictable
    </p>
</div>
""", unsafe_allow_html=True)