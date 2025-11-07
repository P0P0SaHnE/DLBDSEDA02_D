# import modules ---------------------------------------------------------
import pandas as pd


# main code --------------------------------------------------------------
df = pd.read_csv("costumer_complain_data/consumer_complaints.csv")

# check the data file for values and show the first five rows 
print("\n",df.shape,"\n")
print(df.columns,"\n")
print(df.head(5))

df.isnull().sum()
df = df.dropna(subset=["consumer_complaint_narrative"])
print("\n",df.shape,"\n")

df = df.drop_duplicates()
print("\n",df.shape,"\n")
