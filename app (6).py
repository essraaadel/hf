import joblib
import pandas as pd
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import tempfile

st.title("🔍 Predict Heart Disease !")

# Upload model
model_file = st.file_uploader("Upload a trained model (.pkl or .h5)", type=["pkl", "h5"])

# Upload dataset for schema (optional)
schema_file = st.file_uploader("Upload a dataset to extract input features", type=["csv"])

if model_file and schema_file:
    df = pd.read_csv(schema_file)

    if 'HeartDisease' in df.columns:
        df = df.drop('HeartDisease', axis=1)

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    st.subheader("✍️ Enter Input Values")

    input_data = {}
    for col in numeric_cols:
        val = float(df[col].mean())
        input_data[col] = st.number_input(col, value=val)
    for col in categorical_cols:
        options = df[col].dropna().unique().tolist()
        input_data[col] = st.selectbox(col, options)

    input_df = pd.DataFrame([input_data])

    if st.button("🔮 Predict"):
        try:
            if model_file.name.endswith(".pkl"):
                model = joblib.load(model_file)

                # If model includes pipeline with preprocessing
                y_pred = model.predict(input_df)[0]
                st.success(f"🧠 Prediction: {'Heart Disease' if y_pred == 1 else 'No Heart Disease'}")

            elif model_file.name.endswith(".h5"):
                # TEMP save keras model
                with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
                    tmp.write(model_file.getbuffer())
                    model = load_model(tmp.name)

                # Build preprocessor (MUST match how model was trained)
                preprocessor = ColumnTransformer([
                    ("num", Pipeline([
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler())
                    ]), numeric_cols),
                    ("cat", Pipeline([
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('encoder', OneHotEncoder(handle_unknown='ignore'))
                    ]), categorical_cols)
                ])

                X_processed = preprocessor.fit_transform(df)  # fit on full data
                input_processed = preprocessor.transform(input_df)
                pred = (model.predict(input_processed) > 0.5).astype("int32")[0][0]

                st.success(f"🧠 Prediction: {'Heart Disease' if pred == 1 else 'No Heart Disease'}")
            else:
                st.error("Unsupported model format.")
        except Exception as e:
            st.error(f"❌ Error: {e}")
else:
    st.info("📂 Please upload both a model file and a dataset for input reference.")
