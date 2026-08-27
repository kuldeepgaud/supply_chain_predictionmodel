# import data manipulation 
import pandas as pd 
import logging
from pathlib import Path
# load the preprocessed dataset
def load_data():
    logging.info('=====data ingestion started======')

    BASE_DIR = Path(__file__).resolve().parents[2]

    data_path = BASE_DIR / "data" / "SCM.csv"


    df = pd.read_csv(r'/Users/kuldeepgaud/supply_chain_predictionmodel/data/SCM.csv')
    logging.info(f'======df.shape{df.shape}======')
    logging.info('=====data ingestion done')
    return df
