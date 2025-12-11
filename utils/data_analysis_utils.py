import pandas as pd

def inspect_data(df, name):
    print(f"\n--- Inspecting {name} ---")
    print(df.head())
    print(f"\nInfo for {name}:")
    print(df.info())
    print(f"\nDescription for {name}:")
    print(df.describe())

def check_missing_values(df, name):
    print(f"\n--- Missing Values in {name} ---")
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_df = pd.DataFrame({'Missing Count': missing, 'Percent': missing_percent})
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values(by='Percent', ascending=False)
    print(missing_df.head(20)) # Show top 20 columns with missing values
    print(f"\nTotal columns with missing values: {len(missing_df)}")

def check_class_distribution(y, name):
    print(f"\n--- Class Distribution in {name} ---")
    print(y.value_counts())
    print(y.value_counts(normalize=True))
