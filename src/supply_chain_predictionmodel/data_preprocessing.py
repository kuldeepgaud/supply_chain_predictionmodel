# import data manipulation library
import pandas as pd
# import preprocessing libraries
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler,FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from scipy.stats.mstats import winsorize
import logging

# start preprocessing
def preprocessing(df):
    logging.info('=======preprocessing started=========')

    # 1. Clean column names(removes the extra spaces in col names)
    logging.info('=======Removed extra spaces from features names========')
    df.columns = df.columns.str.strip()

    # 2. Remove duplicate rows
    logging.info('=======Removed Duplicates========')
    df = df.drop_duplicates()

    # 3. Define X and y
    X = df.drop(columns=['product_wg_ton','Ware_house_ID','WH_Manager_ID','wh_est_year'])
    logging.info(f'=======dropped unwanted columns:{X.shape}=======')
    y = df['product_wg_ton']
    logging.info(f'======seprated y target column:{y.shape}========')

    # 4. Split the dataset into training and testing - (seen:unseen)
    logging.info('=======sperated X and y to avoid data leakage==========')
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42
    )

    # 5. Identify categorical and numerical columns
    logging.info('========segregated categorical column========')
    categorical_data = X.select_dtypes(include=['object']).columns
    logging.info('========segregated numerical column========')
    numerical_data = X.select_dtypes(exclude=['object']).columns

    # 6. Numerical preprocessing
    logging.info('=======built numerical pipeline==========')
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ("winsor",FunctionTransformer(lambda X:winsorize(X,limits=[0.05,0.05],axis=0))),
        ('scaler', MinMaxScaler())
    ])

    # 7. Categorical preprocessing
    logging.info('============built categorical pipeline=========')
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore',sparse_output=False))])

    # 8. Create ColumnTransformer
    logging.info('========transformer created=========')
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_data),
        ('cat', categorical_pipeline, categorical_data)
    ])

    # 9. Fit and transform training data
    logging.info("===Fitting preprocessor on training data and transforming X_train===")
    X_train_processed = preprocessor.fit_transform(X_train)

    # 10. Transform test data
    logging.info("===Transforming X_test using the fitted preprocessor===")
    X_test_processed = preprocessor.transform(X_test)

    logging.info(
    "Preprocessing completed successfully. X_train: %s, X_test: %s, "
    "y_train: %s, y_test: %s.",
    X_train_processed.shape,
    X_test_processed.shape,
    y_train.shape,
    y_test.shape
)
    return (X_train_processed,X_test_processed,y_train, y_test,preprocessor)


