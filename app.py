from scipy import spatial
import streamlit as st
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import time
from statsmodels.tsa.arima.model import ARIMA
import folium
from folium.plugins import HeatMap, MarkerCluster
from rtree import index
import os
from datetime import datetime, timedelta
import plotly.express as px
import matplotlib.colors as mcolors
import streamlit.components.v1 as components
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import pydeck as pdk
import warnings
from streamlit_option_menu import option_menu

warnings.filterwarnings("ignore")

# Configuration
CITY_MUNICIPALITY = "Echague"
PROVINCE = "Isabela" 
COUNTRY = "Philippines"
BACKUP_FILE_PATH = "C:\\Users\\jakeq\\OneDrive - isu.edu.ph\\BSCS\\4TH YEAR\\THESIS WRITING 2\\DATASETS\\Barangay.csv"
BACKUP_DATA = None 
GEOLOCATOR = Nominatim(user_agent=f"PNP_{CITY_MUNICIPALITY}_Crime_App")

# Clustering parameters
eps_spatial = 0.0002
eps_temporal = 432000.0
min_samples = 5

# Page config
st.set_page_config(page_title="Crime Mapping System", layout="wide")

# ============================================================================
# LOGIN SYSTEM INTEGRATION
# ============================================================================

# Custom CSS for login page
def inject_login_css():
    st.markdown("""
    <style>
    .main {
        background: transparent;
        color: white;
    }

    .stApp {
        background-image: url('https://as1.ftcdn.net/v2/jpg/06/10/09/38/1000_F_610093815_QvahpZYZVHUs3jXgQxGxrF0TMdmpbeph.jpg');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    h1, h2, h3, h4, h5, h6 {
        color: white !important;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.8); 
    }
    
    .stRadio > label, .stSlider > label, .stSelectbox > label {
        color: white !important;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.8); 
    }

    div[data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.95);
        padding: 2.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 2px solid rgba(255, 255, 255, 0.18);
    }

    .form-title {
        color: #1e3c72 !important;
        text-align: center;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: none !important;
    }

    .subtitle {
        text-align: center;
        color: #555;
        margin-bottom: 2rem;
        font-size: 0.95rem;
    }

    .stTextInput > label {
        color: #1e3c72 !important;
        font-weight: 600;
        text-shadow: none !important;
    }

    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin-top: 1rem;
    }

    .footer {
        text-align: center;
        color: #ffffff;
        margin-top: 2rem;
        font-size: 0.85rem;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.8);
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state for authentication
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None

# Initialize session state for data
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_clustered' not in st.session_state:
    st.session_state.df_clustered = None
if 'clustering_complete' not in st.session_state:
    st.session_state.clustering_complete = False
if 'geocoding_complete' not in st.session_state:
    st.session_state.geocoding_complete = False
if 'forecast_data' not in st.session_state:  # ← ADD THIS
    st.session_state.forecast_data = None
if 'clustering_ran' not in st.session_state:  # ← ADD THIS
    st.session_state.clustering_ran = False
if 'selected_barangays' not in st.session_state:  # ← ADD THIS
    st.session_state.selected_barangays = []

# Authentication function
def authenticate(username, password):
    valid_credentials = {
        "admin": "admin",
        "user": "user"
    }
    return username in valid_credentials and valid_credentials[username] == password

# Login page
def login_page():
    inject_login_css()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<h1 class="form-title">🔒 Law Enforcement Portal</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Secure Access System</p>', unsafe_allow_html=True)
        
        with st.form("login_form", clear_on_submit=True):
            username = st.text_input("👤 Username", placeholder="Enter your username")
            password = st.text_input("🔑 Password", type="password", placeholder="Enter your password")
            
            col_a, col_b, col_c = st.columns([1, 2, 1])
            with col_b:
                submit = st.form_submit_button("🔓 Login", use_container_width=True)
            
            if submit:
                if username and password:
                    if authenticate(username, password):
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.rerun()
                    else:
                        st.markdown('<div class="error-box">❌ Invalid credentials. Access denied.</div>', 
                                unsafe_allow_html=True)
                else:
                    st.markdown('<div class="error-box">⚠️ Please enter both username and password.</div>', 
                            unsafe_allow_html=True)
        
        st.markdown('<div class="footer">🛡️ Authorized Personnel Only<br>All access attempts are logged and monitored</div>', 
                unsafe_allow_html=True)

# ============================================================================
# MAIN APPLICATION FUNCTIONS (from 1.py)
# ============================================================================

def run_disaggregated_arima(df, selected_barangays, forecast_days):
    """
    1. Uses weekly ARIMA to generate a stable total forecast for the horizon.
    2. Distributes that total evenly across the requested forecast_days.
    3. Returns a daily-indexed DataFrame.
    """
    forecast_results = []
    forecast_weeks = int(np.ceil(forecast_days / 7))

    df_filtered = df[df['Barangay'].isin(selected_barangays)].dropna(subset=['timestamp']).copy()
    
    if df_filtered.empty:
        return pd.DataFrame(columns=['Incident_Date', 'Predicted_Count', 'Barangay'])
        
    last_historical_date = df_filtered['timestamp'].max().normalize()
    forecast_dates_daily = pd.date_range(
        start=last_historical_date + timedelta(days=1), 
        periods=forecast_days, 
        freq='D'
    )

    for barangay in selected_barangays:
        brgy_df = df_filtered[df_filtered['Barangay'] == barangay].copy()

        if 'timestamp' in brgy_df.columns:
            brgy_df.set_index('timestamp', inplace=True)

        if len(brgy_df.index.normalize().unique()) < 30: 
            continue

        brgy_weekly = brgy_df.resample('W-SUN').size().rename('Count')
        brgy_weekly.index = pd.to_datetime(brgy_weekly.index)
        brgy_weekly = brgy_weekly.asfreq('W-SUN', fill_value=0)

        try:
            model_brgy = ARIMA(brgy_weekly, order=(1, 0, 1)) 
            results_brgy = model_brgy.fit()
            forecast_result = results_brgy.get_forecast(steps=forecast_weeks)
            forecast_series_weekly = forecast_result.predicted_mean
            total_predicted_weekly = forecast_series_weekly.sum()
            total_predicted_incidents = np.maximum(total_predicted_weekly, 0).round().astype(int)

            daily_baseline = total_predicted_incidents / forecast_days
            daily_forecasts = []
            cumulative_prediction = 0
            
            for date in forecast_dates_daily:
                cumulative_prediction += daily_baseline
                
                if cumulative_prediction >= 1:
                    predicted_count_today = int(np.floor(cumulative_prediction))
                    cumulative_prediction -= predicted_count_today
                else:
                    predicted_count_today = 0
                
                daily_forecasts.append({
                    'Incident_Date': date,
                    'Predicted_Count': predicted_count_today,
                    'Barangay': barangay
                })

            df_forecast = pd.DataFrame(daily_forecasts)
            
            if not df_forecast.empty:
                forecast_results.append(df_forecast)

        except Exception as e:
            continue

    if forecast_results:
        return pd.concat(forecast_results, ignore_index=True)
    else:
        return pd.DataFrame(columns=['Incident_Date', 'Predicted_Count', 'Barangay'])

def load_backup_data(file_path):
    """Loads the backup CSV file for geocoding fallback."""
    global BACKUP_DATA
    try:
        df_backup = pd.read_csv(file_path)
        df_backup['Barangay'] = df_backup['Barangay'].astype(str).str.strip()
        df_backup = df_backup[['Barangay', 'Latitude', 'Longitude']].rename(
            columns={'Latitude': 'LAT_BACKUP', 'Longitude': 'LON_BACKUP'}
        )
        BACKUP_DATA = df_backup.set_index('Barangay').to_dict('index')
        return True
    except FileNotFoundError:
        st.warning(f"Backup file not found at: {file_path}")
        return False
    except Exception as e:
        st.warning(f"Backup file unavailable: {e}")
        return False

def geocode_barangay(barangay_name, city, province, country="Philippines"):
    """Enhanced geocoding with backup file priority."""
    brgy_name_clean = str(barangay_name).strip()

    if BACKUP_DATA and brgy_name_clean in BACKUP_DATA:
        coords = BACKUP_DATA[brgy_name_clean]
        if pd.notna(coords['LAT_BACKUP']) and pd.notna(coords['LON_BACKUP']):
            return coords['LAT_BACKUP'], coords['LON_BACKUP']

    time.sleep(1.2)
    
    queries = [
        f"Barangay {brgy_name_clean}, {city}, {province}, {country}",
        f"{brgy_name_clean}, {city}, {province}, {country}",
        f"{brgy_name_clean}, {city}, {province}",
        f"{brgy_name_clean}, Echague",
    ]

    for query in queries: 
        try:
            location = GEOLOCATOR.geocode(query, timeout=10)
            if location:
                lat = float(location.latitude)
                lon = float(location.longitude)
                
                if 16.5 <= lat <= 16.9 and 121.5 <= lon <= 121.9:
                    return lat, lon
        except (ValueError, Exception):
            continue
            
    return None, None

def batch_geocode_dataset(df):
    """Applies geocoding to unique Barangay names."""
    load_backup_data(BACKUP_FILE_PATH)

    unique_barangays = df['Barangay'].dropna().unique()
    total_barangays = len(unique_barangays)

    geo_results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, brgy in enumerate(unique_barangays):
        brgy_name = str(brgy).strip()
        progress = (idx + 1) / total_barangays
        progress_bar.progress(progress)
        status_text.text(f"Geocoding: {brgy_name} ({idx + 1}/{total_barangays})")

        lat, lon = geocode_barangay(brgy_name, CITY_MUNICIPALITY, PROVINCE, COUNTRY)
        geo_results.append({'Barangay': brgy, 'LAT': lat, 'LON': lon})

    progress_bar.empty()
    status_text.empty()

    geo_df = pd.DataFrame(geo_results)
    failed_barangays = geo_df[geo_df['LAT'].isna()]['Barangay'].tolist()
    failed_count = len(failed_barangays)
    
    if failed_barangays:
        st.warning(f"⚠️ Could not geocode {failed_count} barangays: {', '.join(map(str, failed_barangays[:10]))}")

    df = df.drop(columns=['LAT', 'LON'], errors='ignore')
    df_geocoded = pd.merge(df, geo_df, on='Barangay', how='left')
    
    initial_size = len(df_geocoded)
    df_geocoded.dropna(subset=['LAT', 'LON'], inplace=True)
    removed_rows = initial_size - len(df_geocoded)

    return df_geocoded, removed_rows, total_barangays, failed_count

def run_cd_dbscan(df):
    """CD-DBSCAN clustering with R-tree spatial indexing."""
    df = df.reset_index(drop=True)
    
    df['LAT'] = df['LAT'].astype(str).str.replace('[^0-9\.\-]', '', regex=True)
    df['LON'] = df['LON'].astype(str).str.replace('[^0-9\.\-]', '', regex=True)
    df['LAT'] = pd.to_numeric(df['LAT'], errors='coerce')
    df['LON'] = pd.to_numeric(df['LON'], errors='coerce')
    df.dropna(subset=['LAT', 'LON'], inplace=True)
    
    coords = df[['LAT', 'LON']].values
    times = df['seconds'].values
    visited = set()
    labels = [-1] * len(df)
    cluster_id = 0

    spatial_index = index.Index()
    for i, row in df.iterrows():
        spatial_index.insert(i, (row['LON'], row['LAT'], row['LON'], row['LAT']))

    query_times = []

    def get_neighbors(i):
        start_q = time.perf_counter()
        lat, lon, t = coords[i][0], coords[i][1], times[i]
        box = (lon - eps_spatial, lat - eps_spatial, lon + eps_spatial, lat + eps_spatial)
        spatial_candidates = list(spatial_index.intersection(box))
        neighbors = [j for j in spatial_candidates if abs(times[j] - t) <= eps_temporal]
        end_q = time.perf_counter()
        query_times.append((end_q - start_q) * 1000)
        return neighbors

    for i in range(len(df)):
        if i in visited:
            continue
        neighbors = get_neighbors(i)
        if len(neighbors) < min_samples:
            visited.add(i)
            continue
        labels[i] = cluster_id
        seeds = set(neighbors)
        while seeds:
            current = seeds.pop()
            if current not in visited:
                visited.add(current)
                current_neighbors = get_neighbors(current)
                if len(current_neighbors) >= min_samples:
                    seeds.update(current_neighbors)
            if labels[current] == -1:
                labels[current] = cluster_id
        cluster_id += 1

    avg_query_time = np.mean(query_times) if query_times else 0
    df['cluster'] = labels
    return df, cluster_id, avg_query_time

def save_clustered_csv(df):
    """Saves clustered data to history folder."""
    SAVE_FOLDER = 'dbscan_history' 
    os.makedirs(SAVE_FOLDER, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"dbscan_clustered_output_{timestamp}.csv"
    save_path = os.path.join(SAVE_FOLDER, filename)

    try:
        df.to_csv(save_path, index=False)
        return {"status": "success", "path": save_path}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ============================================================================
# MAIN APPLICATION INTERFACE
# ============================================================================

def main_application():
    """Main crime mapping application (after successful login)"""
    
    # Header with logout button
    col1, col2 = st.columns([5, 1])
    with col1:
        st.title("🗺️ Enhance Community Safety with Crime Mapping")
        st.caption(f"👤 Logged in as: **{st.session_state.username}**")
    with col2:
        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.df = None
            st.session_state.df_clustered = None
            st.session_state.clustering_complete = False
            st.rerun()
    
    st.markdown("---")
    
    st.subheader("📂 Upload Records")
    file = st.file_uploader("Upload your CSV file", type=["csv"])
    st.info("Note: Only CSV files are accepted. Required columns: Date_Occ, Crime_Type, Barangay, Time_Occ")

    if file:
        df = pd.read_csv(file)
        
        original_size = len(df)
        required_cols = ['Date_Occ', 'Crime_Type', 'Barangay']
        
        if not all(col in df.columns for col in required_cols):
            st.error(f"Missing required columns: {', '.join(required_cols)}")
            st.stop()
        
        df['timestamp'] = pd.to_datetime(df['Date_Occ'], errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)
        df['seconds'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()
        
        cleaned_size = len(df)
        removed_rows = original_size - cleaned_size
        
        st.success("✅ Data preprocessed successfully!")
        st.write(f"Original: {original_size} rows | Cleaned: {cleaned_size} rows ({removed_rows} removed)")
        
        needs_geocoding = 'LAT' not in df.columns or 'LON' not in df.columns
        
        if needs_geocoding:
            st.warning("⚠️ Latitude/Longitude columns not found. Geocoding required.")
            
            if st.button("🌍 Start Geocoding & Clustering"):
                with st.spinner("Step 1/2: Geocoding barangays..."):
                    df_geocoded, removed, total_brgy, failed = batch_geocode_dataset(df)
                    
                    if df_geocoded.empty:
                        st.error("Geocoding failed for all barangays. Cannot proceed.")
                        st.stop()
                    
                    st.success(f"Geocoding complete: {total_brgy - failed}/{total_brgy} successful")
                    st.session_state.geocoding_complete = True
                
                with st.spinner("Step 2/2: Running CD-DBSCAN clustering..."):
                    df_clustered, clusters_found, avg_query = run_cd_dbscan(df_geocoded)
                    
                    st.session_state.df_clustered = df_clustered
                    st.session_state.clustering_complete = True
                    
                    noise_points = (df_clustered['cluster'] == -1).sum()
                    clustered_points = len(df_clustered) - noise_points
                    
                    st.success("✅ Clustering complete!")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Clusters Found", clusters_found)
                    col2.metric("Clustered Points", f"{clustered_points:,}")
                    col3.metric("Noise Points", noise_points)
                    col4.metric("Query Time (ms)", f"{avg_query:.2f}")
        else:
            st.info("✅ Dataset already contains LAT/LON columns.")
            
            if st.button("🚀 Run CD-DBSCAN Clustering"):
                with st.spinner("Running clustering..."):
                    df_clustered, clusters_found, avg_query = run_cd_dbscan(df)
                    
                    st.session_state.df_clustered = df_clustered
                    st.session_state.clustering_complete = True
                    
                    noise_points = (df_clustered['cluster'] == -1).sum()
                    clustered_points = len(df_clustered) - noise_points
                    
                    st.success("✅ Clustering complete!")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Clusters Found", clusters_found)
                    col2.metric("Clustered Points", f"{clustered_points:,}")
                    col3.metric("Noise Points", noise_points)
                    col4.metric("Query Time (ms)", f"{avg_query:.2f}")

    # Menu (only show if clustering is complete)
    if st.session_state.clustering_complete:
        st.markdown("---")
        
        menu = option_menu(
            "Menu",
            ["Visualizations", "Forecasting", "Reports", "Download Output"],
            icons=['map', 'graph-up', 'file-text', 'download'],
            menu_icon="cast",
            default_index=0,
            orientation="horizontal",
            styles={
                "container": {"padding": "5px", "background-color": "#d7deeb"},
                "icon": {"color": "black", "font-size": "20px"}, 
                "nav-link": {
                    "font-size": "18px",
                    "text-align": "left",
                    "margin":"2px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "#AF84E1"},
            }
        )
        df = st.session_state.df_clustered

        if menu == "Visualizations":
            st.header("🗺️ Visualizations")
                    # 1. Prepare data for Cluster Visualizations (Valid LAT/LON and assigned cluster != -1)
            df_clustered = df[(df['LAT'] != 0) & (df['LON'] != 0) & (df['cluster'] != -1)].copy()

            if df_clustered.empty:
                st.warning("No valid, clustered data available for visualization.")
            else:
                st.subheader("Crime Cluster Visualization")

                # User Choice for Map Type
                map_type = st.radio(
                    "Select Cluster Map Type:",
                    ('Density Map', 'Scatter Plot'),
                    horizontal=True
                )

                # Use a sample for better performance, common to both Plotly maps
                df_sample = df_clustered.sample(min(25000, len(df_clustered)))

                # 2. Map Rendering based on User Choice

                if map_type == 'Density Map':
                    st.caption("Shows the concentration of crime clusters.")

                    fig = px.density_mapbox(
                        df_sample.sample(min(10000, len(df_sample))), # Smaller sample is often better for Density
                        lat='LAT',
                        lon='LON',
                        z='cluster', # Use cluster as the density value
                        radius=15,
                        center=dict(lat=df_clustered['LAT'].mean(), lon=df_clustered['LON'].mean()),
                        zoom=10,
                        mapbox_style="open-street-map",
                        title="Crime Cluster Density Map",
                        color_continuous_scale='Viridis'
                    )

                elif map_type == 'Scatter Plot':
                    # Scatter Plot is configured for a single color
                    st.caption("Shows individual crime points in a single color.")

                    fig = px.scatter_mapbox(
                        df_sample,
                        lat='LAT',
                        lon='LON',
                        hover_data=['cluster'],
                        center=dict(lat=df_clustered['LAT'].mean(), lon=df_clustered['LON'].mean()),
                        zoom=10,
                        mapbox_style="open-street-map",
                        title="Crime Cluster Scatter Plot (Single Color)",
                        opacity=0.8
                    )

                    # Set a fixed color and size for all markers
                    fig.update_traces(marker=dict(color='blue', size=5)) 

                # Display the Plotly figure (common to both map types)
                fig.update_layout(height=600, margin={"r":0,"t":50,"l":0,"b":0})
                st.plotly_chart(fig, use_container_width=True)

            # ---------------------------------------------------------------------
            # B. INTERACTIVE HEAT MAP (All Valid Crimes)
            # ---------------------------------------------------------------------

            st.subheader("Interactive Heat Map (All Valid Crimes)")
            # Filter for all valid crime locations (requires only valid LAT/LON)
            df_all_valid = df[(df['LAT'] != 0) & (df['LON'] != 0)].copy()

            if not df_all_valid.empty:
                map_center = [df_all_valid['LAT'].mean(), df_all_valid['LON'].mean()]
                m = folium.Map(location=map_center, zoom_start=12)
                # Use the data frame containing all valid crime locations
                heat_data = [[row['LAT'], row['LON']] for _, row in df_all_valid.iterrows()]
                HeatMap(heat_data).add_to(m)
                components.html(m._repr_html_(), height=600)
            else:
                st.write("No valid data available for the Heat Map.")
                
            st.markdown("---")

            # ---------------------------------------------------------------------
            # C. RISK ASSESSMENT ANALYSIS (Forecast and Pydeck)
            # ---------------------------------------------------------------------

            st.subheader("🎯 Risk Assessment Analysis")

            if df is None or 'cluster' not in df.columns:
                st.warning("Please run clustering in Step 3 first to proceed with risk assessment analysis.")
                # st.stop() # Use st.stop() in your actual app if this is mandatory

            required_cols = ['LAT', 'LON', 'Barangay', 'timestamp']
            if not all(col in df.columns for col in required_cols):
                st.error(f"Missing one or more required columns: {required_cols}.")
                # st.stop() # Use st.stop() in your actual app if this is mandatory
            else:
                st.subheader("Parameters and Selection (Municipality: Echague)")

                all_barangays = sorted(df['Barangay'].dropna().unique().tolist())
                selected_barangays = all_barangays
                
                st.info(f"Analysis will automatically cover **all {len(selected_barangays)} barangays** of Echague.")

                forecast_days = st.slider(
                    "Forecast Horizon (Days)", 
                    min_value=7,
                    max_value=60, 
                    value=30, 
                    step=1
                )

                if not selected_barangays:
                    st.error("No barangay data available to run the predictive analysis.")
                else:
                    # Check if a forecast was run and if the button is clicked
                    if st.button(f"Generate Municipality-Wide Forecast ({forecast_days} Days) 📈"):
                        with st.spinner(f"Running disaggregated ARIMA forecast for {len(selected_barangays)} barangays over {forecast_days} days..."):

                            # --- SIMULATED FORECASTING (Placeholder) ---
                            if 'run_disaggregated_arima' in globals():
                                # If your ARIMA function is defined globally, use it here
                                forecast_data = run_disaggregated_arima(df, selected_barangays, forecast_days)
                            else:
                                # Placeholder/simulated data generation (based on previous request)
                                dates = pd.to_datetime(pd.date_range(start=pd.Timestamp.now(), periods=forecast_days, freq='D'))
                                np.random.seed(42)
                                all_data = []
                                for b in selected_barangays:
                                    # Higher risk for two barangays
                                    counts = np.random.randint(0, 5, size=forecast_days) + (5 if 'San Isidro' in b or 'Victoria' in b else 0)
                                    counts[counts < 0] = 0 # Ensure counts are non-negative
                                    
                                    for i, date in enumerate(dates):
                                        all_data.append({
                                            'Incident_Date': date,
                                            'Predicted_Count': counts[i],
                                            'Barangay': b
                                        })
                                forecast_data = pd.DataFrame(all_data)
                            # --- END SIMULATED FORECASTING ---

                        st.session_state.forecast_data = forecast_data
                        st.session_state.clustering_ran = True # Assuming a placeholder state update
                        st.session_state.selected_barangays = selected_barangays
                        
                        if not forecast_data.empty:
                            st.success(f"Forecast complete for all {len(selected_barangays)} Barangays.")
                        else:
                            st.warning("Forecast completed, but no data was generated. This may be due to insufficient data for daily ARIMA or fitting errors.")

                    # --- DISPLAY AND RISK VISUALIZATION ---
                    if st.session_state.forecast_data is not None and not st.session_state.forecast_data.empty:
                        
                        df_f = st.session_state.forecast_data.copy()
                        df_f['Incident_Date'] = pd.to_datetime(df_f['Incident_Date']) 
                        
                        # Data Aggregation for Risk Scoring
                        df_risk = df_f.groupby('Barangay')['Predicted_Count'].sum().reset_index()
                        
                        # Get mean coordinates for each barangay (for map)
                        df_coords = df[['Barangay', 'LAT', 'LON']].groupby('Barangay').mean().reset_index()
                        df_risk = pd.merge(df_risk, df_coords, on='Barangay', how='inner')

                        # --- Risk Alert Logic ---
                        st.subheader("Accident Risk Assessment Alert 🚨 (Municipality-Wide)")
                        N = 4
                        df_high_risk = df_risk.sort_values(by='Predicted_Count', ascending=False).head(N)

                        if df_high_risk['Predicted_Count'].sum() == 0 and not df_risk.empty:
                            st.info(f"No high-risk barangays detected, as all predicted incident counts are zero for the next {forecast_days} days.")
                            high_risk_barangays = []
                        else:
                            high_risk_barangays = df_high_risk[df_high_risk['Predicted_Count'] > 0]['Barangay'].tolist()

                        if high_risk_barangays:
                            alert_html = f"""
                            <div style="border: 4px solid red; padding: 20px; background-color: #ffe6e6; border-radius: 10px;">
                                <h2 style="color: red; text-align: center;">Alert: Accident Risk Assessment</h2>
                                <p style="color: black; text-align: center;">High accident risk detected in the following barangays:</p>
                                <ul style="color: red; font-weight: bold; list-style-type: none; text-align: center; padding: 0;">
                                    {''.join([f"<li>{b}</li>" for b in high_risk_barangays])}
                                </ul>
                                <br>
                                <p style="color: black; text-align: center;">Recommended Actions:</p>
                                <ul style="color: red; list-style-type: disc; margin: 0 auto; width: fit-content;">
                                    <li>Increase patrols in high-risk areas.</li>
                                    <li>Install additional warning signs or lights.</li>
                                    <li>Consider road infrastructure improvements in these areas.</li>
                                </ul>
                            </div>
                            """
                            st.markdown(alert_html, unsafe_allow_html=True)
                        else:
                            st.info("The forecast did not identify any barangays with a high enough predicted incident count to trigger a high-risk alert.")
                        
                        st.markdown("---")

                        # --- Risk Level Coloring ---
                        
                        df_risk['Color_R'] = 0
                        df_risk['Color_G'] = 0
                        df_risk['Color_B'] = 0
                        df_risk['Risk_Level'] = 'Low Risk' 

                        # Default: Low Risk (Green, where count > 0 is handled below)
                        df_risk.loc[df_risk['Predicted_Count'] >= 0, 'Color_G'] = 255
                        df_risk.loc[df_risk['Predicted_Count'] == 0, 'Risk_Level'] = 'Low Risk'

                        # Moderate Risk: Orange
                        moderate_mask = (~df_risk['Barangay'].isin(high_risk_barangays)) & (df_risk['Predicted_Count'] > 0)
                        df_risk.loc[moderate_mask, ['Color_R', 'Color_G', 'Color_B']] = [255, 165, 0]
                        df_risk.loc[moderate_mask, 'Risk_Level'] = 'Moderate Risk'

                        # High Risk: Red
                        high_mask = df_risk['Barangay'].isin(high_risk_barangays)
                        df_risk.loc[high_mask, ['Color_R', 'Color_G', 'Color_B']] = [255, 0, 0]
                        df_risk.loc[high_mask, 'Risk_Level'] = 'High Risk'

                        # --- Pydeck Visualization (High-Risk Only) ---
                        st.subheader(f"Pydeck Risk Visualization ({forecast_days} Days) 🚨 (High-Risk Barangays Only)")

                        df_high_risk_only = df_risk[df_risk['Risk_Level'] == 'High Risk'].copy()
                        
                        if not df_high_risk_only.empty and df_high_risk_only['Predicted_Count'].sum() > 0:

                            max_count = df_high_risk_only['Predicted_Count'].max()
                            ELEVATION_SCALE = 5000 / (max_count if max_count > 0 else 1)

                            view_state = pdk.ViewState(
                                latitude=df_high_risk_only['LAT'].mean(),
                                longitude=df_high_risk_only['LON'].mean(),
                                zoom=11.5,
                                pitch=50,
                                bearing=-20 
                            )
                            
                            # Column Layer (Cylinders)
                            column_layer = pdk.Layer(
                                'ColumnLayer',
                                data=df_high_risk_only, 
                                get_position='[LON, LAT]',
                                get_elevation='Predicted_Count * ' + str(ELEVATION_SCALE), 
                                elevation_scale=1, 
                                radius=300, 
                                radius_scale=1,
                                get_fill_color='[Color_R, Color_G, Color_B, 220]', 
                                pickable=True,
                                extruded=True,
                                auto_highlight=True,
                            )
                            
                            # Text Layer (Barangay Labels)
                            text_layer = pdk.Layer(
                                'TextLayer',
                                data=df_high_risk_only,
                                get_position='[LON, LAT]',
                                get_text='Barangay',
                                get_size=12,
                                get_color='[255, 255, 255, 255]',
                                # Place text slightly above the column base
                                get_elevation='Predicted_Count * ' + str(ELEVATION_SCALE), 
                                get_angle=0,
                                get_text_anchor='"middle"',
                                get_alignment_baseline='"center"',
                                get_pixel_offset=[0, 30],
                                pickable=False,
                                billboard=True,
                                background=True, 
                                get_background_color='[0, 0, 0, 150]',
                            )

                            tooltip = {
                                "html": "<b>Barangay:</b> {Barangay}<br/><b>Predicted Incidents:</b> {Predicted_Count:.0f}<br/><b>Risk Level:</b> {Risk_Level}",
                                "style": {"backgroundColor": "red", "color": "white"}
                            }

                            r = pdk.Deck(
                                layers=[column_layer, text_layer],
                                initial_view_state=view_state,
                                tooltip=tooltip,
                                map_style='mapbox://styles/mapbox/streets-v11',
                            )

                            st.pydeck_chart(r)
                            
                            # Legend and Explanation
                            st.markdown(f"""
                            **Risk Legend for the Next {forecast_days} Days (Showing High-Risk Only):**
                            <div style="display: flex; gap: 20px; font-weight: bold;">
                                <span style="color: red;">&#9679; Red: Top {N} High Risk (Highest Predicted Incidents)</span>
                            </div>
                            """, unsafe_allow_html=True)

                            st.markdown(f"The **height** of the cylinders represents the **sum of predicted incidents** for each high-risk barangay. Taller cylinders indicate higher risk/volume.")
                            
                        else:
                            st.info("No barangays currently meet the criteria for a High-Risk (Red) alert.")
            
        elif menu == "Forecasting":
            st.header("📈 Forecasting")

            df['month'] = df['timestamp'].dt.to_period('M')
            cluster_counts_monthly = df.groupby('month').size().rename('Count')
            cluster_counts_monthly.index = cluster_counts_monthly.index.to_timestamp()
            
            if not cluster_counts_monthly.empty:
                try:
                    model = ARIMA(cluster_counts_monthly, order=(1, 1, 1))
                    results = model.fit()
                    forecast = results.forecast(steps=7)

                    df_observed = cluster_counts_monthly.reset_index()
                    df_observed.columns = ['Date', 'Count']
                    df_observed['Type'] = 'Observed'

                    df_forecast = forecast.reset_index()
                    df_forecast.columns = ['Date', 'Count']
                    df_forecast['Type'] = 'Forecast'

                    df_plot = pd.concat([df_observed, df_forecast])

                    chart_type = st.radio("Select Chart Type:", ('Line', 'Bar'), horizontal=True)
                    
                    if chart_type == 'Line':
                        fig2 = px.line(df_plot, x='Date', y='Count', color='Type',
                                    title="Crime Incident Forecast (Monthly)", markers=True)
                    else:
                        fig2 = px.bar(df_plot, x='Date', y='Count', color='Type',
                                    title="Crime Incident Forecast (Monthly)", barmode='group')
                    
                    st.plotly_chart(fig2, use_container_width=True)
                except Exception as e:
                    st.error(f"Forecasting error: {e}")
            
        elif menu == "Reports":
            st.header("📊 Reports")
            
            # 1. TOP 10 CRIME TYPES
            st.subheader("🔴 Top 10 Crime Types")
            
            if 'Crime_Type' in df.columns:
                crime_type_counts = df.groupby('Crime_Type').size().reset_index(name='Count')
                crime_type_counts = crime_type_counts.sort_values('Count', ascending=False).head(10)
                crime_type_counts.insert(0, 'No.', range(1, len(crime_type_counts) + 1))
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.write("**Crime Type Statistics**")
                    st.dataframe(
                        crime_type_counts.style.set_properties(**{'text-align': 'left'}),
                        use_container_width=True,
                        hide_index=True
                    )
                
                with col2:
                    fig_crime = px.bar(
                        crime_type_counts, 
                        x='Count', 
                        y='Crime_Type',
                        orientation='h',
                        title="Top 10 Crime Types",
                        labels={'Crime_Type': 'Crime Type', 'Count': 'Number of Incidents'},
                        color='Count',
                        color_continuous_scale='Reds'
                    )
                    fig_crime.update_layout(yaxis={'categoryorder':'total ascending'}, height=400)
                    st.plotly_chart(fig_crime, use_container_width=True)
            else:
                st.warning("Crime_Type column not found in dataset.")
            
            st.markdown("---")
            
            # 2. TOP 10 BARANGAYS
            st.subheader("📍 Top 10 Barangays with Most Crimes")
            
            if 'Barangay' in df.columns:
                barangay_counts = df.groupby('Barangay').size().reset_index(name='Count')
                barangay_counts = barangay_counts.sort_values('Count', ascending=False).head(10)
                barangay_counts.insert(0, 'No.', range(1, len(barangay_counts) + 1))
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.write("**Barangay Crime Statistics**")
                    st.dataframe(
                        barangay_counts.style.set_properties(**{'text-align': 'left'}),
                        use_container_width=True,
                        hide_index=True
                    )
                
                with col2:
                    fig_barangay = px.bar(
                        barangay_counts, 
                        x='Count', 
                        y='Barangay',
                        orientation='h',
                        title="Top 10 Barangays by Crime Count",
                        labels={'Barangay': 'Barangay', 'Count': 'Number of Incidents'},
                        color='Count',
                        color_continuous_scale='Blues'
                    )
                    fig_barangay.update_layout(yaxis={'categoryorder':'total ascending'}, height=400)
                    st.plotly_chart(fig_barangay, use_container_width=True)
            else:
                st.warning("Barangay column not found in dataset.")
            
            st.markdown("---")
            
            # 3. TOP 10 TIME PERIODS
            st.subheader("🕐 Top 10 Time Periods When Crimes Occur Most")
            
            if 'Time_Occ' in df.columns:
                df_temp = df.copy()
                df_temp['Time_Occ'] = pd.to_datetime(df_temp['Time_Occ'], format='%H:%M:%S', errors='coerce').dt.time
                df_temp['Hour'] = pd.to_datetime(df_temp['Time_Occ'].astype(str), format='%H:%M:%S', errors='coerce').dt.hour
                
                time_counts = df_temp.groupby('Hour').size().reset_index(name='Count')
                time_counts = time_counts.sort_values('Count', ascending=False).head(10)
                time_counts['Time'] = time_counts['Hour'].apply(lambda x: f"{int(x):02d}:00")
                time_counts = time_counts[['Time', 'Count']].reset_index(drop=True)
                time_counts.insert(0, 'No.', range(1, len(time_counts) + 1))
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.write("**Time Period Crime Statistics**")
                    st.dataframe(
                        time_counts.style.set_properties(**{'text-align': 'left'}),
                        use_container_width=True,
                        hide_index=True
                    )
                
                with col2:
                    fig_time = px.bar(
                        time_counts, 
                        x='Count', 
                        y='Time',
                        orientation='h',
                        title="Top 10 Time Periods by Crime Count",
                        labels={'Time': 'Time Period', 'Count': 'Number of Incidents'},
                        color='Count',
                        color_continuous_scale='Greens'
                    )
                    fig_time.update_layout(yaxis={'categoryorder':'total ascending'}, height=400)
                    st.plotly_chart(fig_time, use_container_width=True)
            else:
                st.warning("Time_Occ column not found in dataset.")
            
            st.markdown("---")
            
            # 4. PIE CHARTS
            st.subheader("📊 Distribution Overview")

            if 'Crime_Type' in df.columns:
                crime_pie_data = df.groupby('Crime_Type').size().reset_index(name='Count')
                crime_pie_data = crime_pie_data.sort_values('Count', ascending=False).head(10)
                
                fig_crime_pie = px.pie(
                    crime_pie_data,
                    values='Count',
                    names='Crime_Type',
                    title="Crime Type Distribution (Top 10)",
                    hole=0.3
                )
                fig_crime_pie.update_traces(textposition='inside', textinfo='percent+label')
                fig_crime_pie.update_layout(height=500)
                st.plotly_chart(fig_crime_pie, use_container_width=True)

            if 'Barangay' in df.columns:
                barangay_pie_data = df.groupby('Barangay').size().reset_index(name='Count')
                barangay_pie_data = barangay_pie_data.sort_values('Count', ascending=False).head(10)
                
                fig_barangay_pie = px.pie(
                    barangay_pie_data,
                    values='Count',
                    names='Barangay',
                    title="Barangay Distribution (Top 10)",
                    hole=0.3
                )
                fig_barangay_pie.update_traces(textposition='inside', textinfo='percent+label')
                fig_barangay_pie.update_layout(height=500)
                st.plotly_chart(fig_barangay_pie, use_container_width=True)

            st.markdown("---")
            
            # 5. ALL BARANGAYS OVERVIEW
            st.subheader("🏘️ Overall Crime Incidents by Barangay")
            st.markdown("This bar chart displays the **total number of reported incidents** for **all** available barangays, sorted alphabetically. Use the horizontal scrollbar to view all barangays.")
            
            if 'Barangay' not in df.columns:
                st.error("Missing 'Barangay' column in the dataset.")
            else:
                barangay_counts_all = df.dropna(subset=['Barangay']).groupby('Barangay').size().reset_index(name='Total Incidents')
                barangay_counts_all = barangay_counts_all.sort_values(by='Barangay', ascending=True).reset_index(drop=True)
                
                if barangay_counts_all.empty:
                    st.warning("No valid 'Barangay' data found to generate the bar chart.")
                else:
                    num_barangays = len(barangay_counts_all)
                    st.info(f"✅ Found **{num_barangays}** unique barangays. Use the scrollbar below the chart to view the complete list.")
                    
                    bar_width = 50
                    chart_width = max(1200, num_barangays * bar_width)
                    chart_height = 500
                    
                    fig_barangay_all = px.bar(
                        barangay_counts_all,
                        x='Barangay',
                        y='Total Incidents',
                        color='Barangay',
                        title="Total Crime Counts per Barangay (Alphabetical)",
                        labels={'Barangay': 'Barangay Name', 'Total Incidents': 'Total Crime Incidents'},
                        height=chart_height,
                        width=chart_width
                    )
                    
                    fig_barangay_all.update_layout(
                        xaxis={
                            'tickangle': -45,
                            'title': 'Barangay Name'
                        },
                        yaxis={
                            'title': 'Total Incidents',
                            'tickformat': ',d'
                        },
                        showlegend=False,
                        margin=dict(l=80, r=40, t=80, b=150),
                        yaxis_gridcolor='lightgray'
                    )
                    st.plotly_chart(fig_barangay_all, use_container_width=False)
            
            st.markdown("---")
            
        elif menu == "Download Output":
            st.header("💾 Download Output")
            
            st.write("Preview of clustered data:")
            st.dataframe(df.head(100))
            
            if st.button("💾 Save to Output History"):
                result = save_clustered_csv(df)
                if result['status'] == 'success':
                    st.success(f"✅ File saved: {result['path']}")
                else:
                    st.error(f"❌ Error: {result['message']}")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    if st.session_state.logged_in:
        main_application()
    else:
        login_page()

if __name__ == "__main__":
    main()