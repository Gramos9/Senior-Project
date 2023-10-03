import pandas as pd
import os

# Specify the folder path
folder_path = '/Users/christianrobertson/Desktop/Senior project CSv'

# Get a list of all CSV files in the folder
all_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.csv')]

# Loop through each file, display its name, and print the first few rows of its content
for file in all_files:
    print("\nDisplaying data for:", file)
    df = pd.read_csv(os.path.join(folder_path, file))
    print(df.head())
    print("-" * 50)