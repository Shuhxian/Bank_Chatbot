import pandas as pd

data = pd.read_excel('Database.xlsx', sheet_name = None)

for key in data:
     print(data[key].head())
