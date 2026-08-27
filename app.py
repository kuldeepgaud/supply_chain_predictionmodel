import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

from src.supply_chain_predictionmodel.data_ingestion import load_data
from src.supply_chain_predictionmodel.data_preprocessing import preprocessing


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Supply Chain Prediction",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f7f9fc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        color: #6b7280;
        margin-bottom: 30px;
    }

    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    .prediction-card {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 15px;
        border: 2px solid #e5e7eb;
        text-align: center;
        margin-top: 20px;
    }

    .prediction-value {
        font-size: 42px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "model.pkl"


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def get_data():

    df = load_data()

    df.columns = df.columns.str.strip()

    return df


# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource
def train_model():

    df = get_data()

    X_train, X_test, y_train, y_test, preprocessor = preprocessing(df)

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=3,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    mae = mean_absolute_error(y_test, y_test_pred)

    return (
        model,
        preprocessor,
        X_train,
        X_test,
        y_train,
        y_test,
        train_r2,
        test_r2,
        mae
    )


# ============================================================
# LOAD EVERYTHING
# ============================================================

try:

    df = get_data()

    (
        model,
        preprocessor,
        X_train,
        X_test,
        y_train,
        y_test,
        train_r2,
        test_r2,
        mae
    ) = train_model()

except Exception as e:

    st.error("Unable to load the model or dataset.")

    st.exception(e)

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🚚 Supply Chain")

st.sidebar.markdown(
    """
    ### Navigation

    Use the menu below to explore the application.
    """
)

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Dashboard",
        "🔮 Prediction",
        "📊 Analytics",
        "ℹ️ About"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    **Machine Learning Model**

    Random Forest Regressor

    Target:
    `product_wg_ton`
    """
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="main-title">🚚 Supply Chain Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Machine Learning powered supply chain forecasting dashboard</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "📦 Total Records",
            f"{len(df):,}"
        )

    with col2:

        st.metric(
            "📊 Features",
            f"{df.shape[1]:,}"
        )

    with col3:

        st.metric(
            "🎯 Test R²",
            f"{test_r2:.4f}"
        )

    with col4:

        st.metric(
            "📉 MAE",
            f"{mae:,.2f}"
        )

    st.markdown("---")

    # --------------------------------------------------------
    # MODEL PERFORMANCE
    # --------------------------------------------------------

    st.subheader("🤖 Model Performance")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Training R² Score",
            f"{train_r2:.4f}"
        )

    with col2:

        st.metric(
            "Testing R² Score",
            f"{test_r2:.4f}"
        )

    st.markdown("---")

    # --------------------------------------------------------
    # DATASET PREVIEW
    # --------------------------------------------------------

    st.subheader("📋 Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# ============================================================
# PREDICTION
# ============================================================

elif page == "🔮 Prediction":

    st.markdown(
        '<div class="main-title">🔮 Product Weight Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Enter warehouse and operational information</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # CREATE INPUT DATAFRAME
    # --------------------------------------------------------

    input_data = {}

    feature_columns = [
        col for col in df.columns
        if col not in [
            "product_wg_ton",
            "Ware_house_ID",
            "WH_Manager_ID",
            "wh_est_year"
        ]
    ]

    st.subheader("📦 Warehouse Information")

    # Split features into groups
    categorical_columns = df[feature_columns].select_dtypes(
        include=["object"]
    ).columns.tolist()

    numerical_columns = df[feature_columns].select_dtypes(
        exclude=["object"]
    ).columns.tolist()

    # --------------------------------------------------------
    # CATEGORICAL INPUTS
    # --------------------------------------------------------

    if categorical_columns:

        st.markdown("### 🏭 Categorical Information")

        cat_cols = st.columns(2)

        for i, column in enumerate(categorical_columns):

            unique_values = (
                df[column]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            unique_values = sorted(unique_values)

            with cat_cols[i % 2]:

                if len(unique_values) > 0:

                    input_data[column] = st.selectbox(
                        column,
                        unique_values
                    )

    # --------------------------------------------------------
    # NUMERICAL INPUTS
    # --------------------------------------------------------

    st.markdown("### 📊 Numerical Information")

    num_cols = st.columns(2)

    for i, column in enumerate(numerical_columns):

        min_value = float(df[column].min())
        max_value = float(df[column].max())
        median_value = float(df[column].median())

        with num_cols[i % 2]:

            input_data[column] = st.number_input(
                column,
                min_value=min_value,
                max_value=max_value,
                value=median_value
            )

    st.markdown("---")

    # --------------------------------------------------------
    # PREDICT BUTTON
    # --------------------------------------------------------

    if st.button(
        "🚀 Predict Product Weight",
        use_container_width=True,
        type="primary"
    ):

        try:

            input_df = pd.DataFrame(
                [input_data],
                columns=feature_columns
            )

            # Apply same preprocessing
            processed_input = preprocessor.transform(input_df)

            # Make prediction
            prediction = model.predict(processed_input)[0]

            # Result
            st.markdown(
                f"""
                <div class="prediction-card">

                    <h2>Predicted Product Weight</h2>

                    <div class="prediction-value">
                        {prediction:,.2f} tons
                    </div>

                    <p>
                        Prediction generated using Random Forest Regressor
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.success(
                "Prediction generated successfully!"
            )

        except Exception as e:

            st.error("Prediction failed.")

            st.exception(e)


# ============================================================
# ANALYTICS
# ============================================================

elif page == "📊 Analytics":

    st.markdown(
        '<div class="main-title">📊 Supply Chain Analytics</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Explore the underlying dataset</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # DESCRIPTIVE STATISTICS
    # --------------------------------------------------------

    st.subheader("📈 Descriptive Statistics")

    st.dataframe(
        df.describe().T,
        use_container_width=True
    )

    # --------------------------------------------------------
    # TARGET DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("🎯 Product Weight Distribution")

    st.bar_chart(
        df["product_wg_ton"].value_counts().head(20)
    )

    # --------------------------------------------------------
    # CORRELATION
    # --------------------------------------------------------

    st.subheader("🔗 Correlation Matrix")

    numerical_df = df.select_dtypes(
        include=np.number
    )

    correlation = numerical_df.corr()

    st.dataframe(
        correlation.round(2),
        use_container_width=True
    )

    # --------------------------------------------------------
    # RAW DATA
    # --------------------------------------------------------

    st.subheader("📋 Complete Dataset")

    st.dataframe(
        df,
        use_container_width=True
    )


# ============================================================
# ABOUT
# ============================================================

elif page == "ℹ️ About":

    st.markdown(
        '<div class="main-title">ℹ️ About the Project</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        ## Supply Chain Prediction Model

        This application uses Machine Learning to predict
        product weight in a supply-chain environment.

        ### Machine Learning Pipeline

        **1. Data Ingestion**

        Load the supply-chain dataset.

        **2. Data Preprocessing**

        - Remove duplicate records
        - Remove unwanted columns
        - Handle missing values
        - Winsorization
        - Min-Max scaling
        - One-hot encoding

        **3. Model**

        Random Forest Regressor.

        **4. Prediction**

        The trained model predicts:

        `product_wg_ton`

        ### Model Configuration

        - Algorithm: Random Forest Regressor
        - Estimators: 100
        - Maximum Depth: 3
        - Random State: 42

        ### Technologies

        - Python
        - Pandas
        - NumPy
        - Scikit-learn
        - Streamlit
        - MLflow

        ### Developer

        **Kuldeep Gaud**
        """
    )

    st.markdown("---")

    st.markdown(
        "🔗 [View Project on GitHub](https://github.com/kuldeepgaud/supply_chain_predictionmodel)"
    )