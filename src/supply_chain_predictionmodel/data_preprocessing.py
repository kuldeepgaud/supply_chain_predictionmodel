# import data manipulation library
import pandas as pd
# import preprocessing libraries
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler,FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from scipy.stats.mstats import winsorize


# start preprocessing
def preprocessing(df):

    # 1. Clean column names(removes the extra spaces in col names)
    df.columns = df.columns.str.strip()

    # 2. Remove duplicate rows
    df = df.drop_duplicates()

    # 3. Define X and y
    X = df.drop(columns=['product_wg_ton'])
    y = df['product_wg_ton']

    # 4. Split the dataset into training and testing - (seen:unseen)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42
    )

    # 5. Identify categorical and numerical columns
    categorical_data = X.select_dtypes(include=['object']).columns
    numerical_data = X.select_dtypes(exclude=['object']).columns

    # 6. Numerical preprocessing
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ("winsor",FunctionTransformer(lambda X:winsorize(X,limits=[0.05,0.05],axis=0))),
        ('scaler', MinMaxScaler())
    ])

    # 7. Categorical preprocessing
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore',sparse_output=False))])

    # 8. Create ColumnTransformer
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_data),
        ('cat', categorical_pipeline, categorical_data)
    ])

    # 9. Fit and transform training data
    X_train_processed = preprocessor.fit_transform(X_train)

    # 10. Transform test data
    X_test_processed = preprocessor.transform(X_test)


    return (X_train_processed,X_test_processed,y_train, y_test,preprocessor)


