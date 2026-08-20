# import data manipulation library
import pandas as pd
# import preprocessing libraries
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,MinMaxScaler,RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer,KNNImputer
from scipy.stats.mstats import winsorize 

# start preprocessing 
def preprocessing(df):

    # drop duplicate values
    df.drop_duplicates()

    # define X and y features 
    X = df.drop(['product_wg_ton'],inplace = True)
    y = df['product_wg_ton']

    # split the training and testing data
    X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                    test_size= 0.3,
                                                    random_state=1)

    # handles null values and encode the categorical columns
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())])

    # handles the missing values for categorical and encode it
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False
        ))
    ])



    return X_train,X_test,y_train,y_test
