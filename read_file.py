import pandas as pd

df = pd.read_excel(r"Database.xlsx", sheet_name = 'GENERAL_INTENTS', index_col= 0).to_dict(orient='index')
df2 = pd.read_excel(r"Database.xlsx", sheet_name = 'FAQS', index_col= 0).to_dict(orient='index')
df3 = pd.read_excel(r"Database.xlsx", sheet_name = 'DEFAULT_REPLY', index_col= 0).to_dict(orient='index')

df, df2, df3
#print(f"1: {df} \n2: {df2} \n3: {df3} ")
