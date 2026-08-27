from src.supply_chain_predictionmodel.data_ingestion import load_data
from src.supply_chain_predictionmodel.data_preprocessing import preprocessing
from src.supply_chain_predictionmodel.model_build import model_build
import logging
logging.basicConfig(level=logging.INFO,
                    filename='model.log',
                    filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    force=True)
    
def main():
    logging.info("========== ORCHESTRATOR(entry point) EXECUTION STARTED ==========")
    df = load_data()
    print(df.shape)

    X_train,X_test,y_train,y_test,preprocessor = preprocessing(df)
    print(X_train.shape,y_train.shape)
    report_score = model_build(X_train,X_test,y_train,y_test,preprocessor)
    print(report_score)
    logging.info("========== ORCHESTRATOR EXECUTION COMPLETED SUCCESSFULLY ==========")

main()