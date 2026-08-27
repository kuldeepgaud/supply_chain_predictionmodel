# import data manipulation 
import pandas as pd 
import logging
# load the preprocessed dataset
def load_data():
    logging.info('=====data ingestion started======')
    df = pd.read_csv(r'/Users/kuldeepgaud/supply_chain_predictionmodel/data/SCM.csv')
    logging.info(f'======df.shape{df.shape}======')
    logging.info('=====data ingestion done')
    return df
