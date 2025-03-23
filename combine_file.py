import pandas as pd
import os

# Get the current file directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Read the gene list file2
csv_dir = os.path.join(current_dir, 'dataset/partitions')

def get_data():
    list = pd.read_csv(os.path.join(csv_dir, 'list.1300-1400.expr.csv'))
    return list


# common_genes = get_common_genes_among_files()
# print(get_combined_dataframe(common_genes))
