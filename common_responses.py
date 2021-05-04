import pandas as pd

common_answers = pd.read_excel(r"Database.xlsx", sheet_name = 'COMMON_MESSAGES', index_col= 0).to_dict(orient='index')

def get_common_answer(message):
    pass

if __name__ == '__main__':
    print(common_answers)
