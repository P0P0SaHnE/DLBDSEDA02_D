import pandas as pd

df = pd.read_csv("costumer_complaints_data/consumer_complaints.csv")
print(df.shape)
print(df.columns)
df.head()
