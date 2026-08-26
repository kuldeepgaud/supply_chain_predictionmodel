# import model building libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,r2_score


def model_build(X_train,X_test,y_train,y_test,preprocessor):

    # fit the model on seen data
    model = RandomForestRegressor().fit(X_train,y_train)

    y_pred = model.predict(X_test)

    R2_score = r2_score(y_test,y_pred)
    print("the r2 score is :",R2_score)
    best_features = model.feature_importances_
    print(best_features)

    return model ,y_pred
