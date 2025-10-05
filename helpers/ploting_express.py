import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
# Set the default renderer to 'browser' to ensure plots open in the browser
pio.renderers.default = 'browser'
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load CSV
df = pd.read_csv("./helpers/metrics_consolidated.csv")  # replace with your filename

# Create a mapping dictionary
dataset_mapping = {
    'HEALTHY-BRCA': 'G1',
    'HEALTHY-PRE-BRCA': 'G2',
    'PRE-BRCA-BRCA': 'G3',
    'HEALTHY-MT-WT-BRCA': 'G4',
    'HEALTHY-WT-BRCA': 'G5',
    'HEALTHY-MT-BRCA': 'G6',
    'HEALTHY-PRE-BRCA-BRCA': 'G7',
    'PRE-BRCA-BRCA-MT': 'G8',
    'PRE-BRCA-BRCA-WT': 'G9'
}

# Replace the dataset names
df['Dataset'] = df['Dataset'].replace(dataset_mapping)

# --- 1. Grouped Bar Chart (Accuracy across models for each dataset) ---
fig_bar = px.bar(
    df,
    x="Model",
    y="Accuracy",
    color="Method",
    barmode="group",
    facet_col="Dataset",
    title="Model Accuracy by Dataset and Method"
)
fig_bar.show()

# --- 2. Heatmap (Average metric values per Model/Dataset) ---
metrics = ["Kappa", "Accuracy", "ROC_AUC", "F1_Score", "Sensitivity", "Specificity", "Precision"]
heatmap_data = df.groupby(["Model", "Dataset"])[metrics].mean().reset_index()

fig_heat = px.imshow(
    heatmap_data.pivot(index="Model", columns="Dataset", values="Accuracy"),
    color_continuous_scale="Viridis",
    title="Accuracy Heatmap (Models vs Datasets)"
)
fig_heat.show()

# --- 3. Radar Chart (compare metrics for a single Model & Dataset) ---
model_choice = df["Model"].unique()[0]
dataset_choice = df["Dataset"].unique()[0]

radar_data = df[(df["Model"] == model_choice) & (df["Dataset"] == dataset_choice)][metrics].mean()

fig_radar = go.Figure()
fig_radar.add_trace(go.Scatterpolar(
    r=radar_data.values,
    theta=metrics,
    fill='toself',
    name=f"{model_choice} on {dataset_choice}"
))
fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    title=f"Radar Chart for {model_choice} on {dataset_choice}"
)
fig_radar.show()

# --- 4. Facet Line Chart (ROC_AUC trend across datasets) ---
fig_line = px.line(
    df,
    x="Dataset",
    y="ROC_AUC",
    color="Model",
    line_dash="Method",
    title="ROC_AUC across Datasets by Model & Method"
)
fig_line.show()
