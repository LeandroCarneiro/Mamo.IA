import os
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import numpy as np

from helpers.datasetHelper import get_samples, split_healthy_data


# Human Activity Recognition dataset with 561 features
# print("Loading Human Activity Recognition dataset...")
# har = fetch_openml('har', version=1, as_frame=True)
# X = har.data.values
# Y = har.target
directory_path = './datasets'
data_health = get_samples(os.path.join(directory_path, 'DT.Healthy.csv'))

# Load the PAN-CANCER-TRANSPOSED.csv data
healthy_cases, prebrca_cases, cancer_cases = split_healthy_data(data_health)


# Combine the data into a single dataframe
# Tag each list of cases
healthy_cases = pd.DataFrame(healthy_cases)
healthy_cases['Tag'] = 'HEALTHY'
prebrca_cases = pd.DataFrame(prebrca_cases)
prebrca_cases['Tag'] = 'PRE-BRCA'
cancer_cases = pd.DataFrame(cancer_cases)
cancer_cases['Tag'] = 'BRCA'

df_cancer = pd.concat([healthy_cases, prebrca_cases, cancer_cases], ignore_index=True) #blood samples
X = df_cancer.iloc[:, :-1].apply(pd.to_numeric, errors='coerce')
Y = df_cancer.iloc[:, -1]

# Handle missing values by replacing them with the minimum value of each column
print("Number of missing values before handling:", X.isna().sum().sum())

# Replace NaN values with minimum value for each column
for column in X.columns:
    min_value = X[column].min(skipna=True)  # Get minimum non-missing value
    X[column] = X[column].fillna(min_value)  # Fill NaN with minimum value
 
print("Number of missing values after handling:", X.isna().sum().sum())

print("X shape:", X.shape)
print("X preview:")
print(X.head())


feature_names = {index: value for index, value in enumerate(df_cancer[0])}
# df = har.data

df_cancer.info()
f_classif_selector = SelectKBest(score_func=f_classif, k=10)
f_classif_selector.fit(X, Y)
f_classif_scores = f_classif_selector.scores_
f_classif_pvalues = f_classif_selector.pvalues_
f_classif_selected_features = f_classif_selector.get_feature_names_out()
print("F-ANOVA scores:", f_classif_scores)
print("F-ANOVA p-values:", f_classif_pvalues)
print("F-ANOVA selected features:", f_classif_selected_features)


# Get the indices of the top 10 features
top_indices = f_classif_selector.get_support(indices=True)
top_features = np.array([feature_names[i] for i in top_indices])
top_scores = f_classif_scores[top_indices]

# Sort by score for better visualization
sorted_idx = np.argsort(top_scores)[::-1]
sorted_features = top_features[sorted_idx]
sorted_scores = top_scores[sorted_idx]

# Create the plot with Plotly Express
import plotly.express as px


# Create a DataFrame for better plotly integration
plot_df = pd.DataFrame({
    'Feature': sorted_features,
    'F-Score': sorted_scores
})

# Create horizontal bar chart
fig = px.bar(plot_df, 
             x='F-Score', 
             y='Feature',
             orientation='h',
             title='Top 10 Features Selected by F-ANOVA')

# Customize layout
fig.update_layout(
    yaxis={'categoryorder': 'total ascending'},
    xaxis_title='F-Score',
    yaxis_title='Selected Features',
    height=500
)

# Add score labels on the bars
fig.update_traces(
    texttemplate='%{x:.2f}',
    textposition='outside'
)

# Show the plot
fig.show()