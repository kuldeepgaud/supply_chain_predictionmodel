# import model building libraries
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score

import warnings
warnings.filterwarnings("ignore")

import logging
import os
import pickle


def model_build(X_train, X_test, y_train, y_test, Transformer):

    logging.info("========== MODEL BUILDING STARTED ==========")

    # Enable MLflow autologging
    mlflow.sklearn.autolog(log_datasets=False)
    logging.info("========== MLflow autologging enabled ==========")

    # Start MLflow run
    with mlflow.start_run():
        logging.info("========== MLflow run started ==========")

        # Create Random Forest Regressor
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=3,
            random_state=42)

        logging.info(
            "RandomForestRegressor created | "
            "n_estimators=100 | max_depth=3 | random_state=42")

        # Cross-validation
        logging.info("========== CROSS VALIDATION STARTED ==========")
        kf = KFold(n_splits=5,shuffle=True,)

        cv_scores = cross_val_score(model,X_train,y_train,cv=kf,scoring="r2")

        cv_mean = cv_scores.mean()

        print("Cross-validation scores:", cv_scores)
        print("Mean cross-validation R2 score:", cv_mean)

        logging.info(f"Cross-validation scores:{cv_scores}")
        logging.info(f"Mean cross-validation R2 score: {cv_mean}")

        logging.info("========== CROSS VALIDATION COMPLETED ==========")

        # Fit the model on training data
        model.fit(X_train, y_train)

        logging.info("========== MODEL TRAINING COMPLETED ==========")

        # Evaluate model on training and test data
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)

        print(f"Train R2 score: {train_score}")
        print(f"Test R2 score: {test_score}")

        logging.info(
            "Model evaluation completed | "
            f"Train R2: {train_score} | Test R2: {test_score}"
        )

        # Generate predictions
        y_pred = model.predict(X_test)

        logging.info("========== PREDICTION COMPLETED ==========")

        # Calculate evaluation metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        print(f"R2 score: {r2}")
        print(f"Mean Absolute Error: {mae}")

        logging.info(
            "Evaluation metrics generated | "
            f"R2 Score: {r2} | MAE: {mae}",
        )

        # Feature importance
        feature_importance = model.feature_importances_
        print("Feature importances:", feature_importance)

        logging.info("========== FEATURE IMPORTANCE GENERATED ==========")

        # Create artifacts directory
        os.makedirs("artifacts", exist_ok=True)

        logging.info("Artifacts directory created.")

        # Save model
        with open("artifacts/model.pkl", "wb") as f:
            pickle.dump(model, f)

        logging.info(
            "Model pickle file generated | Path: artifacts/model.pkl"
        )

        logging.info("========== MODEL BUILDING COMPLETED ==========")

        return model, y_pred
