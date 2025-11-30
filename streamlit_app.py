import streamlit as st
import pandas as pd
import pickle
def load_model():
    with open("kmeans_wine_quality_k3.pkl", "rb") as f:
        model = pickle.load(f)
    return model

kmeans = load_model()

# -----------------------------
# Group descriptions
# -----------------------------
cluster_descriptions = {
    0: "Cluster 0 wines generally have **lower acidity** and **moderate alcohol**. They tend to be smoother and lighter.",
    1: "Cluster 1 wines usually show **high acidity**, **high sulphates**, and **strong flavor intensity**.",
    2: "Cluster 2 wines are typically **sweeter**, **higher alcohol**, and usually have **better quality ratings**."
}

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🍷 Wine Cluster Classifier (KMeans k=3)")

st.write("Upload wine data or enter values manually to see which group it belongs to.")

# Data upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep=";")
    st.write("### Uploaded Data", df)

    # Remove quality column if present
    X = df.drop(columns=["quality"], errors="ignore")

    # Predict clusters
    preds = kmeans.predict(X)
    df["Cluster"] = preds

    # Add descriptions
    df["Description"] = df["Cluster"].map(cluster_descriptions)

    st.write("### Clustered Data")
    st.dataframe(df)

else:
    st.write("### Manual Input")

    # Create input fields dynamically based on model features
    feature_names = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                     "chlorides", "free sulfur dioxide", "total sulfur dioxide",
                     "density", "pH", "sulphates", "alcohol"]

    user_data = {}

    for feature in feature_names:
        user_data[feature] = st.number_input(f"{feature}", value=0.0)

    if st.button("Predict Cluster"):
        # Convert input to DataFrame
        user_df = pd.DataFrame([user_data])

        cluster = int(kmeans.predict(user_df)[0])
        st.success(f"Predicted Group: **Cluster {cluster}**")
        st.info(cluster_descriptions[cluster])
