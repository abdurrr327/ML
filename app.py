import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, accuracy_score, silhouette_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import date, timedelta

# --- Theme Colors ---
BB_GREEN = "#2ECC71"
BB_YELLOW = "#F1C40F"
BB_DARK_GRAY = "#2C3E50"
BB_WHITE = "#FFFFFF"

# --- Page Configuration ---
st.set_page_config(
    page_title="Heisenberg's Lab",
    page_icon="🧪", # You can use an emoji or provide a path to a favicon file
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Load Custom CSS ---
def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file '{file_name}' not found. Using default styles.")

load_css("style.css")

# --- Asset Paths ---
ASSET_DIR = "assets"
WELCOME_GIF = os.path.join(ASSET_DIR, "welcome.gif")
BEAKER_GIF = os.path.join(ASSET_DIR, "beaker.gif")
FILTER_GIF = os.path.join(ASSET_DIR, "filter.gif")
REACTION_GIF = os.path.join(ASSET_DIR, "reaction.gif")
INSPECT_GIF = os.path.join(ASSET_DIR, "inspect.gif")
ERROR_GIF = os.path.join(ASSET_DIR, "error.gif")

# --- Helper Functions ---
def display_gif(gif_path, caption=None):
    """Displays a GIF if the file exists."""
    if os.path.exists(gif_path):
        st.image(gif_path, caption=caption)
    else:
        st.warning(f"GIF not found at {gif_path}. Please place it in the '{ASSET_DIR}' folder.")

def create_themed_plotly_layout(title):
    """Creates a Plotly layout with Breaking Bad theme."""
    return go.Layout(
        title={'text': title, 'x': 0.5, 'font': {'color': BB_YELLOW, 'size': 20}},
        paper_bgcolor='rgba(0,0,0,0)', # Transparent background
        plot_bgcolor='rgba(44, 62, 80, 0.7)', # Dark gray plot area
        font=dict(color=BB_WHITE),
        xaxis=dict(gridcolor=BB_GREEN, linecolor=BB_YELLOW, zerolinecolor=BB_GREEN, title_font=dict(color=BB_YELLOW), tickfont=dict(color=BB_WHITE)),
        yaxis=dict(gridcolor=BB_GREEN, linecolor=BB_YELLOW, zerolinecolor=BB_GREEN, title_font=dict(color=BB_YELLOW), tickfont=dict(color=BB_WHITE)),
        legend=dict(font=dict(color=BB_WHITE), bordercolor=BB_YELLOW, borderwidth=1)
    )

def handle_lab_accident(message, e=None):
    """Displays a themed error message."""
    st.error(f"💥 Lab Accident! {message}", icon="🔥")
    display_gif(ERROR_GIF)
    if e:
        st.error(f"Technical Details: {e}")

@st.cache_data # Cache data loading
def load_kragle_data(uploaded_file):
    """Loads data from uploaded Kragle CSV."""
    try:
        df = pd.read_csv(uploaded_file)
        # Attempt to parse dates if a 'Date' column exists
        if 'Date' in df.columns:
            try:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date')
            except Exception:
                st.info("Could not automatically parse 'Date' column. Ensure it's in a standard format.")
        elif 'date' in df.columns:
             try:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
             except Exception:
                st.info("Could not automatically parse 'date' column. Ensure it's in a standard format.")
        return df.copy() # Return a copy to avoid modifying cache
    except Exception as e:
        handle_lab_accident("Failed to read the Kragle CSV.", e)
        return None

@st.cache_data # Cache yfinance calls
def fetch_yfinance_data(ticker, start_date, end_date):
    """Fetches stock data from Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        if df.empty:
            st.warning(f"No data found for ticker {ticker} in the specified date range.")
            return None
        return df.copy() # Return a copy
    except Exception as e:
        handle_lab_accident(f"Failed to fetch data for ticker {ticker}.", e)
        return None

# --- Session State Initialization ---
# Use session state to keep track of progress and data across steps
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
if 'data' not in st.session_state:
    st.session_state.data = None
if 'data_source' not in st.session_state:
    st.session_state.data_source = None
if 'preprocessed_data' not in st.session_state:
    st.session_state.preprocessed_data = None
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = None
if 'target_variable' not in st.session_state:
    st.session_state.target_variable = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'model_type' not in st.session_state:
    st.session_state.model_type = "Linear Regression" # Default model

# --- Welcome Interface ---
st.title("🧪 Heisenberg's Financial Data Lab 🧪")
st.markdown("## Cooking Up ML Models")
display_gif(WELCOME_GIF, caption="Welcome to the Lab...")

# --- Sidebar - Lab Inventory ---
with st.sidebar:
    st.header("⚗️ Lab Inventory ⚗️")

    # Data Source Selection
    data_source_option = st.radio(
        "Select Your Raw Material:",
        ("Upload Kragle Dataset (CSV)", "Fetch Yahoo Finance Data"),
        key="data_source_selector",
        help="Choose where to get the financial data."
    )
    st.session_state.data_source = data_source_option

    # Kragle Upload
    if st.session_state.data_source == "Upload Kragle Dataset (CSV)":
        uploaded_file = st.file_uploader(
            "Upload your Kragle CSV Formula:",
            type=["csv"],
            help="Upload a financial dataset in CSV format."
        )
        st.session_state.kragle_file = uploaded_file
        st.session_state.yf_ticker = None # Reset yfinance state
        st.session_state.yf_start = None
        st.session_state.yf_end = None

    # Yahoo Finance Fetch
    elif st.session_state.data_source == "Fetch Yahoo Finance Data":
        ticker = st.text_input(
            "Stock Ticker Formula (e.g., AAPL, GOOGL):",
            value="AAPL",
            help="Enter the stock symbol from Yahoo Finance."
        ).upper()
        # Date Range Selection
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date Formula:",
                date.today() - timedelta(days=365*2), # Default to 2 years ago
                help="Select the start date for historical data."
            )
        with col2:
            end_date = st.date_input(
                "End Date Formula:",
                date.today(), # Default to today
                help="Select the end date for historical data."
            )
        st.session_state.yf_ticker = ticker
        st.session_state.yf_start = start_date
        st.session_state.yf_end = end_date
        st.session_state.kragle_file = None # Reset Kragle state

        # (Bonus) Refresh Button
        if st.button("🔄 Refresh the Formula", key="refresh_yfinance"):
            # Clear cache for yfinance function to force refetch
            fetch_yfinance_data.clear()
            st.success("Attempting to refresh Yahoo Finance data...", icon="⏳")
            # Optionally trigger data loading immediately or prompt user to click "Cook" again
            st.session_state.current_step = 0 # Reset pipeline to reload data
            st.rerun() # Rerun the script to reflect changes


    # (Bonus) Model Selection
    st.markdown("---")
    st.subheader("🧪 Choose Your Formula 🧪")
    model_choice = st.selectbox(
        "Select the ML Synthesis Method:",
        ("Linear Regression", "K-Means Clustering", "Logistic Regression"), # Add more models if implemented
        index=["Linear Regression", "K-Means Clustering", "Logistic Regression"].index(st.session_state.model_type), # Keep previous selection
        key="model_selector",
        help="Pick the machine learning algorithm to apply."
    )
    st.session_state.model_type = model_choice
    st.info(f"Selected Formula: **{st.session_state.model_type}**")


# --- Main Application Area ---

st.markdown("---")
st.header("The ML Cooking Process")

# --- Step 1: Load Data ---
st.subheader("Step 1: Load Raw Materials")
if st.button("🍳 Cook the Data", key="load_data"):
    data_loaded = False
    if st.session_state.data_source == "Upload Kragle Dataset (CSV)":
        if st.session_state.kragle_file is not None:
            with st.spinner("Loading Kragle dataset..."):
                st.session_state.data = load_kragle_data(st.session_state.kragle_file)
                if st.session_state.data is not None:
                    data_loaded = True
        else:
            handle_lab_accident("No Kragle file uploaded. Please upload a CSV in the Lab Inventory.")
    elif st.session_state.data_source == "Fetch Yahoo Finance Data":
        if st.session_state.yf_ticker and st.session_state.yf_start and st.session_state.yf_end:
             with st.spinner(f"Fetching data for {st.session_state.yf_ticker}..."):
                st.session_state.data = fetch_yfinance_data(
                    st.session_state.yf_ticker,
                    st.session_state.yf_start,
                    st.session_state.yf_end
                )
                if st.session_state.data is not None:
                    data_loaded = True
        else:
            handle_lab_accident("Please provide a valid ticker and date range in the Lab Inventory.")

    if data_loaded:
        st.session_state.current_step = 1 # Move to next step
        st.success(f"Data cooked successfully! Shape: {st.session_state.data.shape}", icon="✅")
        display_gif(BEAKER_GIF)
        st.dataframe(st.session_state.data.head())
        # Reset subsequent steps if data is reloaded
        st.session_state.preprocessed_data = None
        st.session_state.selected_features = None
        st.session_state.target_variable = None
        st.session_state.X_train = None
        st.session_state.X_test = None
        st.session_state.y_train = None
        st.session_state.y_test = None
        st.session_state.model = None
        st.session_state.predictions = None
        st.session_state.metrics = None
    else:
        st.session_state.current_step = 0 # Stay at current step if loading failed

# Only show next steps if data is loaded successfully
if st.session_state.current_step >= 1 and st.session_state.data is not None:
    st.markdown("---")
    st.subheader("Step 2: Data Preprocessing")
    if st.button("💧 Purify the Mixture", key="preprocess"):
        with st.spinner("Purifying data... handling missing values..."):
            df_processed = st.session_state.data.copy()

            # --- Handle Missing Values ---
            # Select numeric columns for imputation
            numeric_cols = df_processed.select_dtypes(include=np.number).columns
            categorical_cols = df_processed.select_dtypes(exclude=np.number).columns

            missing_before = df_processed.isnull().sum()
            missing_before = missing_before[missing_before > 0]

            if not missing_before.empty:
                st.write("Missing values before purification:")
                st.dataframe(missing_before.to_frame(name='Missing Count'))

                # Simple Imputation (Mean for numeric, Mode for categorical)
                if not df_processed[numeric_cols].empty:
                    num_imputer = SimpleImputer(strategy='mean')
                    df_processed[numeric_cols] = num_imputer.fit_transform(df_processed[numeric_cols])

                if not df_processed[categorical_cols].empty:
                    cat_imputer = SimpleImputer(strategy='most_frequent')
                    df_processed[categorical_cols] = cat_imputer.fit_transform(df_processed[categorical_cols])

                missing_after = df_processed.isnull().sum().sum()
                st.write(f"Missing values after purification: {missing_after}")
                if missing_after == 0:
                     st.info("Missing values handled using mean/mode imputation.", icon="ℹ️")
                else:
                     st.warning("Some missing values might still remain.", icon="⚠️")

            else:
                st.info("No missing values detected in the dataset.", icon="ℹ️")

            # --- (Optional) Handle Outliers (Example using IQR - uncomment if needed) ---
            # st.write("Handling outliers using IQR method (optional)...")
            # for col in numeric_cols:
            #     Q1 = df_processed[col].quantile(0.25)
            #     Q3 = df_processed[col].quantile(0.75)
            #     IQR = Q3 - Q1
            #     lower_bound = Q1 - 1.5 * IQR
            #     upper_bound = Q3 + 1.5 * IQR
            #     # Cap outliers
            #     df_processed[col] = np.where(df_processed[col] < lower_bound, lower_bound, df_processed[col])
            #     df_processed[col] = np.where(df_processed[col] > upper_bound, upper_bound, df_processed[col])
            # st.info("Outliers capped at 1.5*IQR range.", icon="🎯")

            # --- (Optional) Scaling (Example - uncomment and choose scaler if needed) ---
            # Note: Scaling is often done *after* train/test split, especially for supervised learning
            # If using K-Means, scaling *before* is common.
            if st.session_state.model_type == "K-Means Clustering":
                st.write("Scaling numeric features for K-Means using StandardScaler...")
                scaler = StandardScaler()
                df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
                st.info("Numeric features scaled.", icon="📏")

            st.session_state.preprocessed_data = df_processed
            st.session_state.current_step = 2
            st.success("Mixture purified successfully!", icon="✨")
            display_gif(FILTER_GIF)
            st.dataframe(st.session_state.preprocessed_data.head())


if st.session_state.current_step >= 2 and st.session_state.preprocessed_data is not None:
    st.markdown("---")
    st.subheader("Step 3: Feature Engineering & Selection")

    # Feature Selection UI
    all_columns = st.session_state.preprocessed_data.columns.tolist()
    numeric_columns = st.session_state.preprocessed_data.select_dtypes(include=np.number).columns.tolist()

    # Default features/target based on model type
    default_features = []
    default_target = None
    if st.session_state.model_type == "Linear Regression":
        # Common for stock data: use OHLC to predict Close or Adj Close
        potential_features = ['Open', 'High', 'Low', 'Volume']
        potential_targets = ['Close', 'Adj Close']
        default_features = [col for col in potential_features if col in numeric_columns]
        default_target = next((col for col in potential_targets if col in numeric_columns), None)
    elif st.session_state.model_type == "K-Means Clustering":
        # K-Means is unsupervised, no target. Often visualize with 2 features.
        default_features = numeric_columns[:2] # Select first two numeric columns by default
        default_target = None
    elif st.session_state.model_type == "Logistic Regression":
        # Needs a categorical target. Let's try creating one based on price change.
        potential_features = ['Open', 'High', 'Low', 'Volume']
        target_col_options = ['Close', 'Adj Close']
        valid_target_cols = [col for col in target_col_options if col in numeric_columns]
        if valid_target_cols:
            target_choice = valid_target_cols[0] # Use Close or Adj Close
            # Create a binary target: 1 if price increased from previous day, 0 otherwise
            st.session_state.preprocessed_data['Price_Change'] = st.session_state.preprocessed_data[target_choice].diff()
            st.session_state.preprocessed_data['Target_Direction'] = (st.session_state.preprocessed_data['Price_Change'] > 0).astype(int)
            st.session_state.preprocessed_data = st.session_state.preprocessed_data.dropna(subset=['Price_Change']) # Drop first row NaN
            default_target = 'Target_Direction'
            default_features = [col for col in potential_features if col in numeric_columns]
            numeric_columns = st.session_state.preprocessed_data.select_dtypes(include=np.number).columns.tolist() # Update numeric columns
            all_columns = st.session_state.preprocessed_data.columns.tolist() # Update all columns
            st.info(f"Created binary target '{default_target}' based on daily change in '{target_choice}'.", icon="📈")
        else:
             handle_lab_accident("Cannot create binary target for Logistic Regression. Requires 'Close' or 'Adj Close' column.")
             default_target = None # Prevent proceeding
             default_features = []

    # Let user override defaults
    selected_features = st.multiselect(
        "Select Feature Columns (Ingredients):",
        options=numeric_columns, # Primarily use numeric features for these models
        default=default_features,
        key="feature_selector",
        help="Choose the input columns (X) for the model."
    )

    target_variable = None
    if st.session_state.model_type in ["Linear Regression", "Logistic Regression"]:
        # Filter possible target columns (numeric for Linear, potentially pre-created binary for Logistic)
        possible_targets = numeric_columns if st.session_state.model_type == "Linear Regression" else [default_target] if default_target else []
        if default_target and default_target not in possible_targets: # Ensure default_target is valid
            possible_targets.append(default_target)
        
        # Exclude selected features from target options if they overlap
        possible_targets = [col for col in possible_targets if col not in selected_features and col in all_columns]

        if possible_targets:
            target_variable = st.selectbox(
                "Select Target Variable (The Final Product):",
                options=possible_targets,
                index=possible_targets.index(default_target) if default_target in possible_targets else 0,
                key="target_selector",
                help="Choose the column the model should predict."
            )
        else:
            st.warning("No suitable target variable available based on selected features and model type.")

    if st.button("🧪 Select the Ingredients", key="select_features"):
        is_selection_valid = False
        if not selected_features:
            handle_lab_accident("No features selected. Please choose ingredients for the formula.")
        elif st.session_state.model_type != "K-Means Clustering" and not target_variable:
             handle_lab_accident("No target variable selected. Please choose the final product to synthesize.")
        elif st.session_state.model_type != "K-Means Clustering" and target_variable in selected_features:
             handle_lab_accident("Target variable cannot also be a feature. Choose different columns.")
        else:
            is_selection_valid = True

        if is_selection_valid:
            st.session_state.selected_features = selected_features
            st.session_state.target_variable = target_variable
            st.session_state.current_step = 3

            st.success("Ingredients selected for the formula!", icon="👌")
            st.write("**Selected Features (X):**", st.session_state.selected_features)
            if st.session_state.target_variable:
                st.write("**Target Variable (y):**", st.session_state.target_variable)
            else:
                st.write("**Target Variable (y):** None (Unsupervised)")

            # (Bonus) Feature Importance for Linear Regression (Requires model to be trained first, maybe move this)
            # For now, just confirm selection. We can add importance plot after training.


if st.session_state.current_step >= 3 and st.session_state.selected_features:
    st.markdown("---")
    # --- Step 4: Train/Test Split (Only for Supervised Learning) ---
    if st.session_state.model_type in ["Linear Regression", "Logistic Regression"]:
        st.subheader("Step 4: Data Splitting")
        test_size = st.slider("Select Test Set Size (%):", min_value=10, max_value=50, value=25, step=5, key="test_split_slider")

        if st.button("🔪 Divide the Batch", key="split_data"):
            try:
                X = st.session_state.preprocessed_data[st.session_state.selected_features]
                y = st.session_state.preprocessed_data[st.session_state.target_variable]

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size/100.0, random_state=42 # Use random_state for reproducibility
                )

                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.current_step = 4

                st.success(f"Batch divided successfully! ({100-test_size}% Train / {test_size}% Test)", icon="📊")

                # Visualize split
                split_data = pd.DataFrame({
                    'Set': ['Training Set', 'Testing Set'],
                    'Count': [len(X_train), len(X_test)]
                })
                fig = px.pie(split_data, names='Set', values='Count', title='Train/Test Split Distribution',
                             color_discrete_sequence=[BB_GREEN, BB_YELLOW])
                fig.update_layout(create_themed_plotly_layout('Train/Test Split Distribution'))
                fig.update_traces(textinfo='percent+label', marker=dict(line=dict(color=BB_DARK_GRAY, width=2)))
                st.plotly_chart(fig, use_container_width=True)

                st.write("Training Data Shape:", X_train.shape)
                st.write("Testing Data Shape:", X_test.shape)

            except Exception as e:
                handle_lab_accident("Failed to split the data batch.", e)
                st.session_state.current_step = 3 # Stay at previous step
    else:
        # For K-Means, skip the split step conceptually, move directly to training
        st.session_state.current_step = 4 # Mark step 4 as 'complete' for flow control
        st.info("Skipping Train/Test split for K-Means Clustering (Unsupervised).", icon="ℹ️")


if st.session_state.current_step >= 4:
    st.markdown("---")
    st.subheader("Step 5: Model Training")

    # K-Means specific parameter
    n_clusters = 3 # Default
    if st.session_state.model_type == "K-Means Clustering":
        n_clusters = st.slider("Select Number of Clusters (K):", min_value=2, max_value=10, value=3, step=1, key="kmeans_k")

    if st.button("🔥 Brew the Model", key="train_model"):
        with st.spinner(f"Brewing the {st.session_state.model_type} model... This might take a moment..."):
            try:
                model = None
                if st.session_state.model_type == "Linear Regression":
                    model = LinearRegression()
                    model.fit(st.session_state.X_train, st.session_state.y_train)
                elif st.session_state.model_type == "Logistic Regression":
                    # Scale data for Logistic Regression (often beneficial)
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(st.session_state.X_train)
                    st.session_state.scaler = scaler # Save scaler to transform test set later
                    model = LogisticRegression(random_state=42, solver='liblinear') # Good solver for smaller datasets
                    model.fit(X_train_scaled, st.session_state.y_train)
                    st.info("Features scaled using StandardScaler before Logistic Regression training.", icon="📏")
                elif st.session_state.model_type == "K-Means Clustering":
                    # Use preprocessed (and possibly scaled) data
                    X_cluster = st.session_state.preprocessed_data[st.session_state.selected_features]
                    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # n_init='auto' or 10
                    model.fit(X_cluster)
                    # Add cluster labels back to the preprocessed data for visualization
                    st.session_state.preprocessed_data['Cluster'] = model.labels_

                st.session_state.model = model
                st.session_state.current_step = 5
                st.success(f"{st.session_state.model_type} model brewed to perfection!", icon="🎉")
                display_gif(REACTION_GIF)

                # Display model parameters or info
                if st.session_state.model_type == "Linear Regression":
                    st.write(f"Intercept (beta_0): {model.intercept_:.4f}")
                    coeffs = pd.DataFrame(model.coef_, st.session_state.selected_features, columns=['Coefficient (beta_i)'])
                    st.write("Coefficients:")
                    st.dataframe(coeffs)
                elif st.session_state.model_type == "K-Means Clustering":
                    st.write(f"Number of clusters (K): {model.n_clusters}")
                    st.write(f"Cluster Centers Shape: {model.cluster_centers_.shape}")
                    # st.write("Cluster Centers:") # Can be large, maybe omit or show sample
                    # st.dataframe(pd.DataFrame(model.cluster_centers_, columns=st.session_state.selected_features))
                elif st.session_state.model_type == "Logistic Regression":
                     st.write(f"Classes found: {model.classes_}")
                     # Display coefficients (log-odds ratios)
                     try:
                        coeffs = pd.DataFrame(model.coef_.T, st.session_state.selected_features, columns=['Log-Odds Ratio'])
                        st.write("Coefficients (Log-Odds Ratios):")
                        st.dataframe(coeffs)
                        st.write(f"Intercept: {model.intercept_[0]:.4f}")
                     except Exception as coef_e:
                        st.warning(f"Could not display coefficients: {coef_e}")


            except Exception as e:
                handle_lab_accident(f"Failed to brew the {st.session_state.model_type} model.", e)
                st.session_state.current_step = 4 # Revert step


if st.session_state.current_step >= 5 and st.session_state.model is not None:
    st.markdown("---")
    st.subheader("Step 6: Model Evaluation")

    if st.button("🔬 Test the Product", key="evaluate_model"):
        with st.spinner("Running quality control tests..."):
            try:
                metrics = {}
                predictions = None

                if st.session_state.model_type == "Linear Regression":
                    predictions = st.session_state.model.predict(st.session_state.X_test)
                    rmse = np.sqrt(mean_squared_error(st.session_state.y_test, predictions))
                    metrics['RMSE'] = rmse
                    st.metric(label="Root Mean Squared Error (RMSE)", value=f"{rmse:.4f}")
                    st.info("Lower RMSE indicates better fit for regression models.", icon="📉")
                    st.session_state.predictions = predictions # Save predictions

                elif st.session_state.model_type == "Logistic Regression":
                    # Scale test data using the *same* scaler fitted on training data
                    X_test_scaled = st.session_state.scaler.transform(st.session_state.X_test)
                    predictions = st.session_state.model.predict(X_test_scaled)
                    probabilities = st.session_state.model.predict_proba(X_test_scaled)[:, 1] # Prob of class 1
                    accuracy = accuracy_score(st.session_state.y_test, predictions)
                    metrics['Accuracy'] = accuracy
                    st.metric(label="Accuracy Score", value=f"{accuracy:.4f}")
                    st.info("Accuracy is the percentage of correct predictions for classification.", icon="🎯")
                    st.session_state.predictions = predictions # Save predictions
                    st.session_state.probabilities = probabilities # Save probabilities

                    # Plot Confusion Matrix (Bonus)
                    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
                    cm = confusion_matrix(st.session_state.y_test, predictions, labels=st.session_state.model.classes_)
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=st.session_state.model.classes_)
                    # Need matplotlib figure to display in Streamlit
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots()
                    disp.plot(cmap='viridis', ax=ax) # Themed cmap if desired
                    ax.set_title("Confusion Matrix")
                    st.pyplot(fig)


                elif st.session_state.model_type == "K-Means Clustering":
                    # Evaluation for K-Means often uses Silhouette Score
                    X_cluster = st.session_state.preprocessed_data[st.session_state.selected_features]
                    labels = st.session_state.model.labels_
                    if len(np.unique(labels)) > 1: # Silhouette score requires at least 2 clusters
                        silhouette_avg = silhouette_score(X_cluster, labels)
                        metrics['Silhouette Score'] = silhouette_avg
                        st.metric(label="Silhouette Score", value=f"{silhouette_avg:.4f}")
                        st.info("Silhouette Score ranges from -1 to 1. Higher values (closer to 1) indicate better-defined clusters.", icon="🧩")
                    else:
                        st.warning("Silhouette score cannot be calculated with only one cluster.", icon="⚠️")
                    # No separate 'predictions' array for K-Means in the same sense as supervised learning
                    # Cluster labels are stored in st.session_state.preprocessed_data['Cluster']


                st.session_state.metrics = metrics
                st.session_state.current_step = 6
                st.success("Product quality verified!", icon="🏆")

            except Exception as e:
                handle_lab_accident("Failed during product quality testing.", e)
                st.session_state.current_step = 5 # Revert step


if st.session_state.current_step >= 6:
    st.markdown("---")
    st.subheader("Step 7: Results Visualization")

    if st.button("📊 Inspect the Final Batch", key="visualize_results"):
         with st.spinner("Generating final batch inspection report..."):
            try:
                if st.session_state.model_type == "Linear Regression" and st.session_state.predictions is not None:
                    st.write("#### Actual vs. Predicted Values")
                    results_df = pd.DataFrame({
                        'Actual': st.session_state.y_test,
                        'Predicted': st.session_state.predictions
                    }, index=st.session_state.y_test.index) # Keep original index (e.g., Date)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=results_df.index, y=results_df['Actual'], mode='lines', name='Actual', line=dict(color=BB_GREEN, width=2)))
                    fig.add_trace(go.Scatter(x=results_df.index, y=results_df['Predicted'], mode='lines', name='Predicted', line=dict(color=BB_YELLOW, width=2, dash='dash')))
                    fig.update_layout(create_themed_plotly_layout(f'Actual vs. Predicted {st.session_state.target_variable}'),
                                      xaxis_title='Date' if isinstance(results_df.index, pd.DatetimeIndex) else 'Index',
                                      yaxis_title=st.session_state.target_variable)
                    st.plotly_chart(fig, use_container_width=True)

                    # (Bonus) Chemical Composition Chart (Feature Importance for Linear Regression)
                    if hasattr(st.session_state.model, 'coef_'):
                        importance_df = pd.DataFrame({
                            'Feature': st.session_state.selected_features,
                            'Importance': np.abs(st.session_state.model.coef_) # Use absolute value for magnitude
                        }).sort_values(by='Importance', ascending=False)

                        fig_imp = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                                        title='Chemical Composition (Feature Importance)',
                                        color_discrete_sequence=[BB_GREEN])
                        fig_imp.update_layout(create_themed_plotly_layout('Chemical Composition (Feature Importance)'))
                        fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig_imp, use_container_width=True)


                elif st.session_state.model_type == "K-Means Clustering":
                    st.write("#### Cluster Visualization")
                    df_clustered = st.session_state.preprocessed_data.copy()
                    df_clustered['Cluster'] = df_clustered['Cluster'].astype(str) # For discrete colors

                    if len(st.session_state.selected_features) >= 2:
                        # Visualize using the first two selected features
                        x_axis = st.session_state.selected_features[0]
                        y_axis = st.session_state.selected_features[1]

                        fig = px.scatter(df_clustered, x=x_axis, y=y_axis, color='Cluster',
                                        title=f'K-Means Clusters based on {x_axis} and {y_axis}',
                                        color_discrete_map={ # Custom colors for clusters if needed
                                             "0": BB_GREEN, "1": BB_YELLOW, "2": "#3498DB", # Blue
                                             "3": "#E74C3C", # Red "4": "#9B59B6", # Purple etc.
                                             # Add more colors as needed based on max K
                                         })
                        fig.update_layout(create_themed_plotly_layout(f'Clusters: {x_axis} vs {y_axis}'))
                        fig.update_traces(marker=dict(size=8, line=dict(width=1, color=BB_DARK_GRAY)), selector=dict(mode='markers'))
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Need at least 2 features selected to create a scatter plot for clusters.")
                    st.dataframe(df_clustered.head()) # Show data with cluster labels

                elif st.session_state.model_type == "Logistic Regression" and st.session_state.predictions is not None:
                     st.write("#### Prediction Results (Test Set)")
                     results_df = pd.DataFrame({
                         'Actual Direction': st.session_state.y_test,
                         'Predicted Direction': st.session_state.predictions,
                         'Probability (Class 1)': st.session_state.probabilities
                     }, index=st.session_state.y_test.index)
                     st.dataframe(results_df.head(10)) # Show sample results
                     st.info("Direction: 1 = Price Increased, 0 = Price Decreased/Stayed Same (relative to previous day)", icon="ℹ️")
                     # Could add more visualizations like ROC curve if needed

                st.session_state.current_step = 7
                st.success("Final batch ready for inspection!", icon="👀")
                display_gif(INSPECT_GIF)

            except Exception as e:
                 handle_lab_accident("Failed during results visualization.", e)
                 st.session_state.current_step = 6 # Revert step


if st.session_state.current_step >= 7:
    st.markdown("---")
    # --- (Bonus) Download Results ---
    st.subheader("📦 Package the Final Batch 📦")

    results_available = False
    if st.session_state.model_type == "Linear Regression" and st.session_state.predictions is not None:
        results_df = pd.DataFrame({'Actual': st.session_state.y_test, 'Predicted': st.session_state.predictions}, index=st.session_state.y_test.index)
        results_available = True
        file_name = f"{st.session_state.yf_ticker or 'kragle'}_linreg_predictions.csv"
    elif st.session_state.model_type == "K-Means Clustering" and 'Cluster' in st.session_state.preprocessed_data.columns:
        results_df = st.session_state.preprocessed_data[[*st.session_state.selected_features, 'Cluster']]
        results_available = True
        file_name = f"{st.session_state.yf_ticker or 'kragle'}_kmeans_clusters.csv"
    elif st.session_state.model_type == "Logistic Regression" and st.session_state.predictions is not None:
         results_df = pd.DataFrame({
             'Actual Direction': st.session_state.y_test,
             'Predicted Direction': st.session_state.predictions,
             'Probability (Class 1)': st.session_state.probabilities
         }, index=st.session_state.y_test.index)
         results_available = True
         file_name = f"{st.session_state.yf_ticker or 'kragle'}_logreg_predictions.csv"
    else:
         results_df = pd.DataFrame() # Empty dataframe
         file_name = "results.csv"


    if results_available:
        csv = results_df.to_csv(index=True).encode('utf-8')
        st.download_button(
            label="⬇️ Download Results CSV",
            data=csv,
            file_name=file_name,
            mime='text/csv',
            key='download_results_button'
        )
        st.success(f"Click the button above to download '{file_name}'. Heisenberg approves!", icon="💰")
    else:
        st.info("No downloadable results generated yet. Complete the steps above.", icon="⏳")

# --- Footer ---
st.markdown("---")
st.markdown("<div style='text-align: center; color: #F1C40F;'><em>Say my name... Heisenberg's Lab™</em></div>", unsafe_allow_html=True)
