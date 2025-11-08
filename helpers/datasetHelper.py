import pandas as pd
import os

def get_samples(file_path):
    # Read the Excel file
    df = pd.read_csv(file_path)
    # Transpose the data to iterate through columns instead of rows
    return df.T.values

def split_data(data):
    healthy_mt_cases = []
    healthy_wt_cases = []
    healthy_unk_cases = []
    prebrca_cases = []
    brca_mt_cases = []
    brca_wt_cases = []

    for row in data:
        if row[-1] == 'HEALTHY-MT':
            healthy_mt_cases.append(row[:-1])
        elif row[-1] == 'HEALTHY-WT':
            healthy_wt_cases.append(row[:-1])
        elif row[-1] == 'HEALTHY-UNK':
            healthy_unk_cases.append(row[:-1])
        elif row[-1] == 'PRE-BRCA':
            prebrca_cases.append(row[:-1])  
        elif row[-1] == 'BRCA-MT':
            brca_mt_cases.append(row[:-1])
        elif row[-1] == 'BRCA-WT':
            brca_wt_cases.append(row[:-1])
            
    return healthy_mt_cases, healthy_wt_cases, healthy_unk_cases, prebrca_cases, brca_mt_cases, brca_wt_cases

def split_ovarian_data(data):
    # Initialize lists for each class
    ov_cases = []
    non_ov_cases = []

    for row in data:
        if row[-1] == 'OV':
            ov_cases.append(row[:-1])
        elif row[-1] == 'NON-OV':
            non_ov_cases.append(row[:-1])

    return ov_cases, non_ov_cases

def split_lung_data(data):
    # Initialize lists for each class
    luad_cases = []
    non_luad_cases = []

    for row in data:
        if row[-1] == 'LUAD':
            luad_cases.append(row[:-1])
        elif row[-1] == 'NON-LUAD':
            non_luad_cases.append(row[:-1])

    return luad_cases, non_luad_cases

def split_colon_data(data):
    # Initialize lists for each class
    crc_cases = []
    non_crc_cases = []

    for row in data:
        if row[-1] == 'CRC':
            crc_cases.append(row[:-1])
        elif row[-1] == 'NON-CRC':
            non_crc_cases.append(row[:-1])

    return crc_cases, non_crc_cases

def split_breast_data(data):
    # Initialize lists for each class
    nonbrca_cases = []
    brca_cases = []

    for row in data:
        if row[-1] == 'NON-BRCA':
            nonbrca_cases.append(row[:-1])
        elif row[-1] == 'BRCA':
            brca_cases.append(row[:-1])

    return brca_cases, nonbrca_cases

# directory_path = './datasets'
# data_health = get_samples(os.path.join(directory_path, 'DT.Healthy.csv'))
# data_ovarian = get_samples(os.path.join(directory_path, 'DT.Ovarian.csv'))
# data_lung = get_samples(os.path.join(directory_path, 'DT.Lung.csv'))
# data_colon = get_samples(os.path.join(directory_path, 'DT.Colorectal.csv'))
# data_breast = get_samples(os.path.join(directory_path, 'DT.Mama.csv'))

# healthy_cases, prebrca_cases, cancer_cases = split_healthy_data(data_health)
# brca_cases, nonbrca_cases = split_breast_data(data_breast)
# crc_cases, non_crc_cases = split_colon_data(data_colon)
# luad_cases, non_luad_cases = split_lung_data(data_lung)
# ov_cases, non_ov_cases = split_ovarian_data(data_ovarian)

# # Print the results
# print(f"Number of healthy cases: {len(healthy_cases)}")
# print(f"Number of pre-cancer cases: {len(prebrca_cases)}")
# print(f"Number of breast cancer cases: {len(cancer_cases)}")

# print(f"Number of non-brca cases: {len(nonbrca_cases)}")
# print(f"Number of brca cases: {len(brca_cases)}")
# print(f"Number of OV cases: {len(ov_cases)}")
# print(f"Number of non-OV cases: {len(non_ov_cases)}")
# print(f"Number of LUAD cases: {len(luad_cases)}")
# print(f"Number of non-LUAD cases: {len(non_luad_cases)}")
# print(f"Number of CRC cases: {len(crc_cases)}")
# print(f"Number of non-CRC cases: {len(non_crc_cases)}")
#Output