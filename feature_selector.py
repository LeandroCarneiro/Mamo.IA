import pandas as pd
from sklearn.calibration import LabelEncoder
import xgboost as xgb
from xgboost import plot_importance

import matplotlib.pyplot as plt

# Load the dataset
df_cancer = pd.read_csv('./datasets/PAN-CANCER-FINAL-TRANSPOSED.csv')

# Assuming the last column is the target variable
X = df_cancer.iloc[:, :-1].apply(pd.to_numeric, errors='coerce')
Y = df_cancer.iloc[:, -1]

label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)

# Train an XGBoost model
model = xgb.XGBClassifier()
model.fit(X, Y_encoded)

# Plot feature importance
plot_importance(model)
plt.show()

# Get feature importance scores
importance = model.feature_importances_

# Create a DataFrame for feature importance
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
})

# Sort the DataFrame by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Save the feature importance to a CSV file
feature_importance_df.to_csv('feature_importance.csv', index=False)

print("Feature selection completed and saved to feature_importance.csv")