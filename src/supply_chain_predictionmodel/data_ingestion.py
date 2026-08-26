# import data manipulation 
import pandas as pd 

# load the preprocessed dataset
def load_data():
    df = pd.read_csv(r'/Users/kuldeepgaud/supply_chain_predictionmodel/data/SCM.csv')
    return df
