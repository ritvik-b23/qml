import pandas as pd
df = pd.read_csv('train_test_network.csv', encoding='utf-8')
# df.info()


for col in df.select_dtypes(include='object').columns:
    print(f"{col}: {df[col].nunique()} unique values")

print(df.head())