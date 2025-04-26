import os
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
import pandas as pd
from sklearn import metrics
from helpers.datasetHelper import get_samples, split_breast_data, split_colon_data, split_healthy_data, split_lung_data, split_ovarian_data
from models import MyXGboost
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import shap

directory_path = './datasets'
data_health = get_samples(os.path.join(directory_path, 'DT.Healthy.csv'))
data_ovarian = get_samples(os.path.join(directory_path, 'DT.Ovarian.csv'))
data_lung = get_samples(os.path.join(directory_path, 'DT.Lung.csv'))
data_colon = get_samples(os.path.join(directory_path, 'DT.Colorectal.csv'))
data_breast = get_samples(os.path.join(directory_path, 'DT.Mama.csv'))

# Load the PAN-CANCER-TRANSPOSED.csv data
healthy_cases, prebrca_cases, cancer_cases = split_healthy_data(data_health)
brca_cases, nonbrca_cases = split_breast_data(data_breast)
crc_cases, non_crc_cases = split_colon_data(data_colon)
luad_cases, non_luad_cases = split_lung_data(data_lung)
ov_cases, non_ov_cases = split_ovarian_data(data_ovarian)


# Combine the data into a single dataframe
# Tag each list of cases
healthy_cases = pd.DataFrame(healthy_cases)
healthy_cases['Tag'] = 'Healthy'
prebrca_cases = pd.DataFrame(prebrca_cases)
prebrca_cases['Tag'] = 'PreBRCA'
cancer_cases = pd.DataFrame(cancer_cases)
cancer_cases['Tag'] = 'Cancer'
brca_cases = pd.DataFrame(brca_cases)
brca_cases['Tag'] = 'BRCA'
nonbrca_cases = pd.DataFrame(nonbrca_cases)
nonbrca_cases['Tag'] = 'NonBRCA'
crc_cases = pd.DataFrame(crc_cases)
crc_cases['Tag'] = 'CRC'
non_crc_cases = pd.DataFrame(non_crc_cases)
non_crc_cases['Tag'] = 'NonCRC'
luad_cases = pd.DataFrame(luad_cases)
luad_cases['Tag'] = 'LUAD'
non_luad_cases = pd.DataFrame(non_luad_cases)
non_luad_cases['Tag'] = 'NonLUAD'
ov_cases = pd.DataFrame(ov_cases)
ov_cases['Tag'] = 'OV'
non_ov_cases = pd.DataFrame(non_ov_cases)
non_ov_cases['Tag'] = 'NonOV'

# Combine the data into a single dataframe
# df_cancer = pd.concat([brca_cases, nonbrca_cases, crc_cases, non_crc_cases, luad_cases, non_luad_cases, ov_cases, non_ov_cases], ignore_index=True) #everything except healthy
# df_cancer = pd.concat([brca_cases, healthy_cases, nonbrca_cases, crc_cases, non_crc_cases, luad_cases, non_luad_cases, ov_cases, non_ov_cases], ignore_index=True) #everything including healthy
# df_cancer = pd.concat([healthy_cases, brca_cases, crc_cases, luad_cases, ov_cases], ignore_index=True) #cases of cancer x healthy
#df_cancer = pd.concat([healthy_cases, prebrca_cases, cancer_cases], ignore_index=True) #blood samples
#df_cancer = pd.concat([healthy_cases, prebrca_cases], ignore_index=False) #blood samples
df_cancer = pd.concat([healthy_cases, cancer_cases], ignore_index=True) #blood samples
#df_cancer = pd.concat([prebrca_cases, cancer_cases], ignore_index=True) #blood samples

# Set the first column as the index (cpg sites)
# df_cancer.set_index(df_cancer.columns[0], inplace=True)

# The last column is the target classes
# Ensure all data is numeric
X = df_cancer.iloc[:, :-1].apply(pd.to_numeric, errors='coerce')
Y = df_cancer.iloc[:, -1]

# Create a pair key variable with index and value of the first column
original_feature_names = {index: value for index, value in enumerate(data_breast[0])}
feature_index = np.array(list(original_feature_names.values()))

# Fill missing values with the lowest value of its cpg site
X = X.apply(lambda col: col.fillna(col.min()), axis=0)

# Apply PCA normalization 0 to 1 on X
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=0.95, svd_solver='full')  # Keep 95% of variance
X = pca.fit_transform(X_scaled)

# Use LabelEncoder to encode the target classes
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)

# Find duplicated instances in the dataframe
duplicated_instances = df_cancer[df_cancer.duplicated()]

# Print the duplicated instances
if not duplicated_instances.empty:
    print("Duplicated instances in the dataframe:")
    print(duplicated_instances)
else:
    print("No duplicated instances found in the dataframe.")

# Print the encoded target classes
print("Encoded target classes (Y_encoded):")
print(np.unique(Y_encoded))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y_encoded, test_size=0.3, shuffle=True)

# Apply SMOTE to balance the training instances
smote = SMOTE(sampling_strategy='auto', random_state=None, k_neighbors=10)
X_train, y_train = smote.fit_resample(X_train, y_train)


modes = [
    # {
    #     'Name': 'RandomForest100',
    #     'Model': MyXGboost.RandomForest100(X_train, y_train)
    # },
    # {
    #     'Name': 'RandomForest200',
    #     'Model': MyXGboost.RandomForest200(X_train, y_train)
    # },
    # {
    #     'Name': 'RandomForest300',
    #     'Model': MyXGboost.RandomForest300(X_train, y_train)
    # },
    # {
    #     'Name': 'XGBoost',
    #     'Model': MyXGboost.XGBoost(X_train, y_train)
    # },
    {
        'Name': 'LightGBM',
        'Model': MyXGboost.LightGBM(X_train, y_train)
    },
    # {
    #     'Name': 'AdaBoost',
    #     'Model': MyXGboost.AdaBoost(X_train, y_train)
    # },
    # {
    #     'Name': 'GradientBoosting',
    #     'Model': MyXGboost.GradientBoosting(X_train, y_train)
    # }
]


for m in modes:
    selector = MyXGboost.ga_feature_selection(m['Model'], X_train, y_train)
    # Gather the best features
    best_features = selector.best_features_
    print("Selected features:", [original_feature_names[i] for i in best_features])

    # Use only these features to train your final model
    X_train_selected = X_train[:, best_features]
    X_test_selected = X_test[:, best_features]

    final_model = m['Model'].fit(X_train_selected, y_train)

    explainer = shap.Explainer(final_model, X_train)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test)

    # shap.dependence_plot(feature_index['Model'], shap_values, X_test)
    # # Cross Validation: CV = 10
    # scores = cross_val_score(model, X_test, y_test, cv=10)
    # print(f'scores: {scores}')
    # print(f'Cross Validation: {scores.mean()}')
    
    y_pred = final_model.predict(X_test)
    print(f'Accuracy: {metrics.accuracy_score(y_test, y_pred)}')
    print('Classification Report:')
    print(metrics.classification_report(y_test, y_pred))
    m['Model'] = final_model

# Create a DataFrame to store the results
results = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])

for m in modes:
    model_name = m['Name']
    model = m['Model']
    # Save the model
    model_filename = f'{model_name}_model.pkl'
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)

    y_pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    report = metrics.classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
    
    # Calculate average precision, recall, and f1-score
    precision = np.mean([report[class_name]['precision'] for class_name in label_encoder.classes_])
    recall = np.mean([report[class_name]['recall'] for class_name in label_encoder.classes_])
    f1_score = np.mean([report[class_name]['f1-score'] for class_name in label_encoder.classes_])
   
    for class_name in label_encoder.classes_:
        class_precision = report[class_name]['precision']
        class_recall = report[class_name]['recall']
        class_f1_score = report[class_name]['f1-score']
        
        # Append the results to the DataFrame
        results = pd.concat([results, pd.DataFrame([{
            'Model': model_name,
            'Class': class_name,
            'Accuracy': accuracy,
            'Precision': class_precision,
            'Recall': class_recall,
            'F1-Score': class_f1_score
        }])], ignore_index=True)

    # Get feature importances for the model
    if hasattr(model, 'feature_importances_'):
          # Get feature importance scores
        importance = model.feature_importances_

        # Map feature indices to their original names
        feature_importance_df = pd.DataFrame({
            'Feature': [original_feature_names[i] for i in range(len(importance))],
            'Importance': importance
        })

        # Sort the DataFrame by importance
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        # Save the feature importance to a CSV file
        feature_importance_df.to_csv(model_name + '_feature_importance.csv', index=False)


    
# Print the results
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print(results)