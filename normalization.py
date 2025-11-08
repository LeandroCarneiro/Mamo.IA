import os
import pandas as pd
from combat.pycombat import pycombat


def get_samples(file_path):
    """
    Load samples from CSV file.
    Assumes the CSV has genes as rows and samples as columns.
    Skips the first data row and last row (labels).
    
    Returns tuple: (data_df, labels_series)
    """
    # Read the full CSV
    df = pd.read_csv(file_path, index_col=0, low_memory=False)
    
    # Extract the last row (labels) before removing it
    labels = df.iloc[-1]
    
    # Skip first data row (index 0) and last row (labels)
    df = df.iloc[1:-1]
    
    # Convert to numeric, coercing any errors
    df = df.apply(pd.to_numeric, errors='coerce')
    
    return df, labels

def normalize_with_combat(datasets_list, dataset_names):
    """
    Apply ComBat batch effect correction to multiple datasets.
    
    Parameters:
    -----------
    datasets_list : list of pd.DataFrame
        List of dataframes where each dataframe has genes as rows and samples as columns
    dataset_names : list of str
        Names of the datasets (used as batch labels)
    
    Returns:
    --------
    pd.DataFrame
        Combined and batch-corrected dataframe
    """
    # Combine all datasets
    combined_data = pd.concat(datasets_list, axis=1)
    
    print(f"Combined data shape: {combined_data.shape}")
    
    # Replace NaN values with column mean
    combined_data = combined_data.fillna(combined_data.mean())
    
    print(f"NaN values remaining: {combined_data.isna().sum().sum()}")
    
    # Create batch labels for each sample
    batch = []
    for i, dataset in enumerate(datasets_list):
        batch.extend([dataset_names[i]] * dataset.shape[1])
    
    # Apply ComBat normalization
    # pycombat expects data with genes as rows and samples as columns
    corrected_data = pycombat(data=combined_data, batch=batch)
    
    return corrected_data

if __name__ == "__main__":
    # Define paths
    directory_path_combined = './datasets/GEO'
    
    # Load datasets and labels
    GSE32396_data, GSE32396_labels = get_samples(os.path.join(directory_path_combined, 'GSE32396-HEALTHYxBRCA.csv'))
    GSE57285_data, GSE57285_labels = get_samples(os.path.join(directory_path_combined, 'GSE57285-HEALTHYxBRCA.csv'))
    #GSE58045_data, GSE58045_labels = get_samples(os.path.join(directory_path_combined, 'GSE58045-HEALTHY-UNK.csv'))
    GSE58119_data, GSE58119_labels = get_samples(os.path.join(directory_path_combined, 'GSE58119-HEALTHYxPRE-BRCA.csv'))
    
    datasets_list = [
        GSE32396_data,
        GSE57285_data,
        #GSE58045_data,
        GSE58119_data
    ]
    
    # Combine all labels
    combined_labels = pd.concat([
        GSE32396_labels,
        GSE57285_labels,
        #GSE58045_labels,
        GSE58119_labels
    ])
    
    dataset_names = [
        'GSE32396',
        'GSE57285',
        #'GSE58045',
        'GSE58119'
    ]
    
    print("Loading datasets...")
    print(f"GSE32396: {GSE32396_data.shape}")
    print(f"GSE57285: {GSE57285_data.shape}")
    #print(f"GSE58045: {GSE58045_data.shape}")
    print(f"GSE58119: {GSE58119_data.shape}")
    
    print("\nApplying ComBat normalization...")
    normalized_data = normalize_with_combat(datasets_list, dataset_names)
    
    print(f"\nNormalized data shape: {normalized_data.shape}")
    
    # Add labels as the last row
    combined_labels.name = 'CANCER'
    normalized_data_with_labels = pd.concat([normalized_data, combined_labels.to_frame().T])
    
    # Save normalized data with labels
    output_path = os.path.join(directory_path_combined, 'normalized_combined_data.csv')
    normalized_data_with_labels.to_csv(output_path)
    print(f"\nNormalized data saved to: {output_path}")
    print(f"Final shape (with labels): {normalized_data_with_labels.shape}")
