# milestone1.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import timedelta
import os

st.set_page_config(layout="wide", page_title="Air Quality Data Analyzer")

# -------------------------
# Configuration
# -------------------------
# Define data paths
DATA_PATHS = {
    "raw": {
        "city_day": Path("data/raw/city_day.csv"),
        "city_hour": Path("data/raw/city_hour.csv")
    },
    "processed": {
        "city_day": Path("data/processed/city_day_cleaned_final.csv"),
        "city_hour": Path("data/processed/city_hour_cleaned_final.csv")
    }
}

POLLUTANTS = ["PM2.5", "PM10", "NO2", "O3", "SO2", "CO", "NOx", "NH3", "Benzene", "Toluene", "Xylene", "AQI"]
COLORS = px.colors.qualitative.Dark24

def check_data_files():
    """Check which data files are available"""
    available_files = {"raw": {}, "processed": {}}
    
    for data_type, paths in DATA_PATHS.items():
        for file_type, path in paths.items():
            if path.exists():
                available_files[data_type][file_type] = path
    
    return available_files

def load_and_prepare_data(file_path, dataset_name):
    """Load data from CSV and prepare it for analysis"""
    try:
        df = pd.read_csv(file_path)
        
        # Clean column names
        df.columns = df.columns.str.strip().str.lower()
        
        # Handle date columns
        date_columns = [col for col in df.columns if 'date' in col or 'time' in col]
        if date_columns:
            primary_date_col = date_columns[0]
            df["datetime"] = pd.to_datetime(df[primary_date_col], errors='coerce')
        else:
            st.error(f"No date column found in {dataset_name}")
            return None
        
        return df
        
    except Exception as e:
        st.error(f"Error loading {dataset_name}: {str(e)}")
        return None

def extract_locations(df):
    """Extract available locations from dataframe"""
    location_columns = [col for col in df.columns if 'city' in col.lower() or 'station' in col.lower()]
    if location_columns:
        location_col = location_columns[0]
        locations = sorted(df[location_col].dropna().unique().tolist())
        return locations, location_col
    return ["All Locations"], None

def extract_available_pollutants(df):
    """Extract available pollutants from dataframe columns"""
    available_pollutants = []
    for pollutant in POLLUTANTS:
        possible_names = [
            pollutant.lower(),
            pollutant.lower().replace('.', ''),
            pollutant.lower().replace('.', '_'),
            pollutant
        ]
        for name in possible_names:
            if name in df.columns:
                available_pollutants.append(pollutant)
                break
    return available_pollutants

def get_actual_column_name(df, pollutant):
    """Get the actual column name in the dataframe for a pollutant"""
    possible_names = [
        pollutant.lower(),
        pollutant.lower().replace('.', ''),
        pollutant.lower().replace('.', '_'),
        pollutant
    ]
    for name in possible_names:
        if name in df.columns:
            return name
    return None

def filter_by_date_range(df, start_date, end_date):
    """Filter data by date range"""
    if start_date and end_date and 'datetime' in df.columns:
        mask = (df['datetime'] >= pd.to_datetime(start_date)) & (df['datetime'] <= pd.to_datetime(end_date))
        return df[mask]
    return df

def compute_basic_stats(df, pollutants):
    """Compute basic statistics for comparison"""
    stats = {}
    for pollutant in pollutants:
        actual_col = get_actual_column_name(df, pollutant)
        if actual_col and actual_col in df.columns:
            data = df[actual_col].dropna()
            if len(data) > 0:
                stats[pollutant] = {
                    'count': len(data),
                    'mean': round(data.mean(), 2),
                    'median': round(data.median(), 2),
                    'std': round(data.std(), 2),
                    'min': round(data.min(), 2),
                    'max': round(data.max(), 2),
                    'missing': df[actual_col].isna().sum(),
                    'missing_pct': round((df[actual_col].isna().sum() / len(df)) * 100, 1)
                }
    return stats

def create_single_pollutant_comparison(raw_df, processed_df, pollutant, location, location_col, start_date, end_date):
    """Create detailed comparison for a single pollutant"""
    actual_col = get_actual_column_name(raw_df, pollutant)
    if not actual_col:
        return None
    
    # Filter data by date range
    raw_filtered = filter_by_date_range(raw_df, start_date, end_date)
    processed_filtered = filter_by_date_range(processed_df, start_date, end_date)
    
    # Filter by location
    if location != "All Locations" and location_col:
        raw_filtered = raw_filtered[raw_filtered[location_col] == location]
        processed_filtered = processed_filtered[processed_filtered[location_col] == location]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f'Raw {pollutant} Distribution',
            f'Processed {pollutant} Distribution', 
            f'Raw {pollutant} Over Time',
            f'Processed {pollutant} Over Time'
        )
    )
    
    # Histograms
    fig.add_trace(
        go.Histogram(x=raw_filtered[actual_col].dropna(), name='Raw', nbinsx=30, marker_color='red'),
        row=1, col=1
    )
    fig.add_trace(
        go.Histogram(x=processed_filtered[actual_col].dropna(), name='Processed', nbinsx=30, marker_color='blue'),
        row=1, col=2
    )
    
    # Time series
    if 'datetime' in raw_filtered.columns and 'datetime' in processed_filtered.columns:
        raw_sample = raw_filtered[['datetime', actual_col]].dropna().sort_values('datetime')
        processed_sample = processed_filtered[['datetime', actual_col]].dropna().sort_values('datetime')
        
        if len(raw_sample) > 1000:
            raw_sample = raw_sample.iloc[::len(raw_sample)//1000]
        if len(processed_sample) > 1000:
            processed_sample = processed_sample.iloc[::len(processed_sample)//1000]
        
        fig.add_trace(
            go.Scatter(x=raw_sample['datetime'], y=raw_sample[actual_col], 
                      mode='lines', name='Raw', line=dict(color='red')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=processed_sample['datetime'], y=processed_sample[actual_col], 
                      mode='lines', name='Processed', line=dict(color='blue')),
            row=2, col=2
        )
    
    fig.update_layout(height=600, showlegend=False, 
                     title_text=f"{pollutant} - Raw vs Processed Data Comparison")
    return fig

def create_multi_pollutant_comparison(raw_df, processed_df, pollutants, location, location_col, start_date, end_date):
    """Create multiple pollutant comparison plots"""
    # Filter data by date range
    raw_filtered = filter_by_date_range(raw_df, start_date, end_date)
    processed_filtered = filter_by_date_range(processed_df, start_date, end_date)
    
    # Filter by location
    if location != "All Locations" and location_col:
        raw_filtered = raw_filtered[raw_filtered[location_col] == location]
        processed_filtered = processed_filtered[processed_filtered[location_col] == location]
    
    # Determine layout based on number of pollutants
    num_pollutants = len(pollutants)
    
    if num_pollutants == 1:
        # Single pollutant - use 2x2 layout
        return create_single_pollutant_comparison(raw_df, processed_df, pollutants[0], location, location_col, start_date, end_date)
    
    else:
        # Multiple pollutants - create comprehensive comparison
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Multiple Pollutants - Raw Data (Time Series)',
                'Multiple Pollutants - Processed Data (Time Series)',
                'Missing Data Comparison',
                'Mean Value Comparison'
            )
        )
        
        # Plot 1: Multiple pollutants - Raw data time series
        for i, pollutant in enumerate(pollutants):
            actual_col = get_actual_column_name(raw_filtered, pollutant)
            if actual_col and actual_col in raw_filtered.columns:
                valid_data = raw_filtered[['datetime', actual_col]].dropna().sort_values('datetime')
                if len(valid_data) > 0:
                    # Sample for better performance
                    if len(valid_data) > 500:
                        valid_data = valid_data.iloc[::len(valid_data)//500]
                    
                    fig.add_trace(
                        go.Scatter(x=valid_data['datetime'], y=valid_data[actual_col],
                                  mode='lines', name=f'Raw {pollutant}',
                                  line=dict(color=COLORS[i % len(COLORS)])),
                        row=1, col=1
                    )
        
        # Plot 2: Multiple pollutants - Processed data time series
        for i, pollutant in enumerate(pollutants):
            actual_col = get_actual_column_name(processed_filtered, pollutant)
            if actual_col and actual_col in processed_filtered.columns:
                valid_data = processed_filtered[['datetime', actual_col]].dropna().sort_values('datetime')
                if len(valid_data) > 0:
                    # Sample for better performance
                    if len(valid_data) > 500:
                        valid_data = valid_data.iloc[::len(valid_data)//500]
                    
                    fig.add_trace(
                        go.Scatter(x=valid_data['datetime'], y=valid_data[actual_col],
                                  mode='lines', name=f'Processed {pollutant}',
                                  line=dict(color=COLORS[i % len(COLORS)], dash='dash')),
                        row=1, col=2
                    )
        
        # Plot 3: Missing data comparison
        missing_data = []
        for pollutant in pollutants:
            raw_col = get_actual_column_name(raw_filtered, pollutant)
            processed_col = get_actual_column_name(processed_filtered, pollutant)
            
            if raw_col and processed_col:
                raw_missing_pct = (raw_filtered[raw_col].isna().sum() / len(raw_filtered)) * 100
                processed_missing_pct = (processed_filtered[processed_col].isna().sum() / len(processed_filtered)) * 100
                missing_data.append({
                    'Pollutant': pollutant,
                    'Raw Missing %': round(raw_missing_pct, 1),
                    'Processed Missing %': round(processed_missing_pct, 1)
                })
        
        if missing_data:
            missing_df = pd.DataFrame(missing_data)
            fig.add_trace(
                go.Bar(x=missing_df['Pollutant'], y=missing_df['Raw Missing %'], 
                       name='Raw Missing %', marker_color='red'),
                row=2, col=1
            )
            fig.add_trace(
                go.Bar(x=missing_df['Pollutant'], y=missing_df['Processed Missing %'], 
                       name='Processed Missing %', marker_color='blue'),
                row=2, col=1
            )
        
        # Plot 4: Statistical comparison (Mean values)
        mean_data = []
        for pollutant in pollutants:
            raw_col = get_actual_column_name(raw_filtered, pollutant)
            processed_col = get_actual_column_name(processed_filtered, pollutant)
            
            if raw_col and processed_col:
                raw_mean = raw_filtered[raw_col].mean()
                processed_mean = processed_filtered[processed_col].mean()
                if not pd.isna(raw_mean) and not pd.isna(processed_mean):
                    mean_data.append({
                        'Pollutant': pollutant,
                        'Raw Mean': round(raw_mean, 2),
                        'Processed Mean': round(processed_mean, 2)
                    })
        
        if mean_data:
            mean_df = pd.DataFrame(mean_data)
            fig.add_trace(
                go.Bar(x=mean_df['Pollutant'], y=mean_df['Raw Mean'], 
                       name='Raw Mean', marker_color='orange'),
                row=2, col=2
            )
            fig.add_trace(
                go.Bar(x=mean_df['Pollutant'], y=mean_df['Processed Mean'], 
                       name='Processed Mean', marker_color='green'),
                row=2, col=2
            )
        
        fig.update_layout(height=800, showlegend=True, 
                         title_text=f"Multiple Pollutant Analysis - {location} ({len(pollutants)} pollutants)")
        return fig

def create_data_quality_comparison(raw_stats, processed_stats, pollutants):
    """Create data quality comparison charts"""
    comparison_data = []
    
    for pollutant in pollutants:
        if pollutant in raw_stats and pollutant in processed_stats:
            raw = raw_stats[pollutant]
            processed = processed_stats[pollutant]
            
            comparison_data.append({
                'Pollutant': pollutant,
                'Missing_Raw': raw['missing_pct'],
                'Missing_Processed': processed['missing_pct'],
                'Mean_Raw': raw['mean'],
                'Mean_Processed': processed['mean'],
                'Std_Raw': raw['std'],
                'Std_Processed': processed['std'],
                'Missing_Reduction': raw['missing_pct'] - processed['missing_pct'],
                'Records_Raw': raw['count'],
                'Records_Processed': processed['count']
            })
    
    return pd.DataFrame(comparison_data)

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.header("Data Controls")
    
    # Check available files
    available_files = check_data_files()
    
    # Show file status
    st.subheader("File Status")
    for data_type, files in available_files.items():
        st.write(f"{data_type.title()} Data:")
        if files:
            for file_type, path in files.items():
                st.success(f"{file_type}: {path}")
        else:
            st.error(f"No {data_type} files found")
    
    # Check if we have both raw and processed data
    has_raw = bool(available_files["raw"])
    has_processed = bool(available_files["processed"])
    
    if not has_raw:
        st.error("No raw data files found. Cannot proceed.")
        st.stop()
    
    # Dataset selection
    dataset_type = st.selectbox(
        "Select Dataset", 
        options=list(available_files["raw"].keys()),
        format_func=lambda x: x.replace("_", " ").title()
    )
    
    # Load raw data
    raw_file_path = available_files["raw"][dataset_type]
    raw_df = load_and_prepare_data(raw_file_path, f"Raw {dataset_type}")
    
    if raw_df is None:
        st.error("Failed to load raw data")
        st.stop()
    
    # Load processed data if available
    processed_df = None
    if has_processed and dataset_type in available_files["processed"]:
        processed_file_path = available_files["processed"][dataset_type]
        processed_df = load_and_prepare_data(processed_file_path, f"Processed {dataset_type}")
    
    # Get location information
    locations, location_col = extract_locations(raw_df)
    location = st.selectbox("Select Location", options=locations)
    
    # Date range selection
    st.subheader("Date Range Selection")
    if 'datetime' in raw_df.columns:
        min_date = raw_df['datetime'].min().date()
        max_date = raw_df['datetime'].max().date()
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
        with col2:
            end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)
        
        if start_date > end_date:
            st.error("Start date must be before end date")
            start_date, end_date = min_date, max_date
    else:
        start_date, end_date = None, None
        st.warning("No date column found for date range filtering")
    
    # Pollutant selection
    available_pollutants = extract_available_pollutants(raw_df)
    
    if not available_pollutants:
        st.error("No pollutant columns found in the dataset.")
        st.stop()
    
    pollutants = st.multiselect(
        "Select Pollutants for Analysis", 
        options=available_pollutants,
        default=available_pollutants[:3] if len(available_pollutants) >= 3 else available_pollutants
    )

# -------------------------
# Main layout
# -------------------------
st.title("Air Quality Data Analyzer")
st.markdown("Comparison between Raw and Processed Data")

# Apply date range filtering
raw_filtered = filter_by_date_range(raw_df, start_date, end_date)
if processed_df is not None:
    processed_filtered = filter_by_date_range(processed_df, start_date, end_date)
else:
    processed_filtered = None

# Data overview
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Raw Records", raw_filtered.shape[0])
with col2:
    st.metric("Raw Columns", raw_filtered.shape[1])
with col3:
    if processed_filtered is not None:
        st.metric("Processed Records", processed_filtered.shape[0])
    else:
        st.metric("Processed Records", "Not available")
with col4:
    if processed_filtered is not None:
        st.metric("Processed Columns", processed_filtered.shape[1])
    else:
        st.metric("Processed Columns", "Not available")

# Show date range info
if start_date and end_date:
    st.write(f"Date Range: {start_date} to {end_date}")

# Compute statistics
raw_stats = compute_basic_stats(raw_filtered, pollutants)
if processed_filtered is not None:
    processed_stats = compute_basic_stats(processed_filtered, pollutants)

# Main analysis tabs - Fixed tab logic
if processed_df is not None:
    tab1, tab2, tab3 = st.tabs(["Pollutant Comparison", "Quality Metrics", "Statistical Analysis"])
else:
    tab1, tab2 = st.tabs(["Statistical Analysis", "Raw Data Explorer"])

# Tab content for when processed data is available
if processed_df is not None:
    with tab1:
        st.header("Pollutant Comparison")
        
        if pollutants:
            # Show number of selected pollutants
            st.write(f"Analyzing {len(pollutants)} pollutant(s): {', '.join(pollutants)}")
            
            # Create appropriate comparison based on number of pollutants
            fig = create_multi_pollutant_comparison(raw_df, processed_df, pollutants, location, location_col, start_date, end_date)
            if fig:
                st.plotly_chart(fig, width='stretch', key="main_comparison")
            else:
                st.warning("Cannot create comparison for selected pollutants")
            
            # Show individual pollutant stats in a different way to avoid duplicate IDs
            if len(pollutants) > 1:
                st.subheader("Individual Pollutant Details")
                selected_pollutant = st.selectbox("Select pollutant for detailed view", pollutants, key="detail_pollutant")
                if selected_pollutant:
                    single_fig = create_single_pollutant_comparison(raw_df, processed_df, selected_pollutant, location, location_col, start_date, end_date)
                    if single_fig:
                        st.plotly_chart(single_fig, width='stretch', key=f"single_{selected_pollutant}")
        else:
            st.info("Please select at least one pollutant to see comparisons")

    with tab2:
        st.header("Data Quality Metrics")
        
        if pollutants:
            # Create metrics cards
            comparison_df = create_data_quality_comparison(raw_stats, processed_stats, pollutants)
            if not comparison_df.empty:
                st.dataframe(comparison_df, width='stretch')
                
                # Visualization of improvements
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.bar(comparison_df, x='Pollutant', y=['Missing_Raw', 'Missing_Processed'],
                                title='Missing Data Percentage - Raw vs Processed',
                                barmode='group')
                    st.plotly_chart(fig, width='stretch', key="missing_comparison")
                
                with col2:
                    fig = px.bar(comparison_df, x='Pollutant', y='Missing_Reduction',
                                title='Reduction in Missing Data (%)')
                    st.plotly_chart(fig, width='stretch', key="missing_reduction")
        else:
            st.info("Please select pollutants to see quality metrics")

    with tab3:
        st.header("Statistical Analysis")
        
        if pollutants:
            # Show statistics for raw data
            st.subheader("Raw Data Statistics")
            raw_stats_data = []
            for pollutant in pollutants:
                if pollutant in raw_stats:
                    raw_stats_data.append({
                        'Pollutant': pollutant,
                        'Count': raw_stats[pollutant]['count'],
                        'Mean': raw_stats[pollutant]['mean'],
                        'Median': raw_stats[pollutant]['median'],
                        'Std Dev': raw_stats[pollutant]['std'],
                        'Min': raw_stats[pollutant]['min'],
                        'Max': raw_stats[pollutant]['max'],
                        'Missing %': raw_stats[pollutant]['missing_pct']
                    })
            
            if raw_stats_data:
                st.dataframe(pd.DataFrame(raw_stats_data), width='stretch', key="raw_stats_table")
            else:
                st.warning("No statistics available for raw data with selected pollutants")
            
            # Show statistics for processed data
            st.subheader("Processed Data Statistics")
            processed_stats_data = []
            for pollutant in pollutants:
                if pollutant in processed_stats:
                    processed_stats_data.append({
                        'Pollutant': pollutant,
                        'Count': processed_stats[pollutant]['count'],
                        'Mean': processed_stats[pollutant]['mean'],
                        'Median': processed_stats[pollutant]['median'],
                        'Std Dev': processed_stats[pollutant]['std'],
                        'Min': processed_stats[pollutant]['min'],
                        'Max': processed_stats[pollutant]['max'],
                        'Missing %': processed_stats[pollutant]['missing_pct']
                    })
            
            if processed_stats_data:
                st.dataframe(pd.DataFrame(processed_stats_data), width='stretch', key="processed_stats_table")
            else:
                st.warning("No statistics available for processed data with selected pollutants")
                
            # Show comparison statistics
            st.subheader("Comparison Statistics")
            comparison_data = []
            for pollutant in pollutants:
                if pollutant in raw_stats and pollutant in processed_stats:
                    raw = raw_stats[pollutant]
                    processed = processed_stats[pollutant]
                    
                    comparison_data.append({
                        'Pollutant': pollutant,
                        'Mean Change': f"{(processed['mean'] - raw['mean']) / raw['mean'] * 100:.1f}%" if raw['mean'] != 0 else "N/A",
                        'Missing Reduction': f"{raw['missing_pct'] - processed['missing_pct']:.1f}%",
                        'Records Change': f"{(processed['count'] - raw['count']) / raw['count'] * 100:.1f}%" if raw['count'] != 0 else "N/A"
                    })
            
            if comparison_data:
                st.dataframe(pd.DataFrame(comparison_data), width='stretch', key="comparison_stats")
        else:
            st.info("Please select pollutants to see statistics")

# Tab content for when only raw data is available
else:
    with tab1:
        st.header("Statistical Analysis (Raw Data Only)")
        
        if pollutants:
            # Show statistics for raw data
            st.subheader("Raw Data Statistics")
            raw_stats_data = []
            for pollutant in pollutants:
                if pollutant in raw_stats:
                    raw_stats_data.append({
                        'Pollutant': pollutant,
                        'Count': raw_stats[pollutant]['count'],
                        'Mean': raw_stats[pollutant]['mean'],
                        'Median': raw_stats[pollutant]['median'],
                        'Std Dev': raw_stats[pollutant]['std'],
                        'Min': raw_stats[pollutant]['min'],
                        'Max': raw_stats[pollutant]['max'],
                        'Missing %': raw_stats[pollutant]['missing_pct']
                    })
            
            if raw_stats_data:
                st.dataframe(pd.DataFrame(raw_stats_data), width='stretch', key="raw_only_stats")
            else:
                st.warning("No statistics available for selected pollutants")
                
            # Show data preview
            st.subheader("Data Preview")
            display_cols = ["datetime", location_col] if location_col else ["datetime"]
            for pollutant in pollutants[:5]:  # Show first 5 pollutants
                actual_col = get_actual_column_name(raw_filtered, pollutant)
                if actual_col:
                    display_cols.append(actual_col)
            
            if display_cols:
                st.dataframe(raw_filtered[display_cols].head(10), width='stretch')
        else:
            st.info("Please select pollutants to see statistics")

    with tab2:
        st.header("Raw Data Explorer")
        st.info("Processed data files not found. Showing only raw data analysis.")
        
        if pollutants:
            # Show raw data preview
            display_cols = ["datetime", location_col] if location_col else ["datetime"]
            for pollutant in pollutants[:5]:  # Show first 5 pollutants
                actual_col = get_actual_column_name(raw_filtered, pollutant)
                if actual_col:
                    display_cols.append(actual_col)
            
            st.dataframe(raw_filtered[display_cols].head(15), width='stretch')
            
            # Show dataset info
            st.subheader("Dataset Information")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Total records: {raw_filtered.shape[0]}")
                st.write(f"Total columns: {raw_filtered.shape[1]}")
                st.write(f"Date range: {raw_filtered['datetime'].min()} to {raw_filtered['datetime'].max()}")
            with col2:
                st.write(f"Location: {location}")
                st.write(f"Selected pollutants: {len(pollutants)}")
                st.write(f"Available pollutants: {len(available_pollutants)}")
        else:
            st.info("Please select pollutants to explore data")

# Footer
st.markdown("---")
st.markdown("Air Quality Data Analyzer | Raw vs Processed Data Comparison")