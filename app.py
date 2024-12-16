import os
import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from urllib.parse import urlparse
import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def evaluate_metrics(actual, pred):
    mse = mean_squared_error(actual, pred)
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return mae, mse, r2


if __name__=="__main__":
    ## Data Ingestion---> reading the data
    csv_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    try:
        data = pd.read_csv(csv_url, sep=';')

        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(data, test_size=0.25, random_state=42, shuffle=True)

        # The predicted column is "quality" which is a scalar from [3, 9]
        train_x = train.drop(["quality"], axis=1)
        test_x = test.drop(["quality"], axis=1)
        train_y = train[["quality"]]
        test_y = test[["quality"]]
    except Exception as e:
        logger.exception("Unable to download training & test CSV, check your internet connection. Error: %s", e)
        sys.exit(1)
        
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():
        # Execute ElasticNet
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        model.fit(train_x, train_y)

        predicted_qualities = model.predict(test_x)

        (mae, mse, r2) = evaluate_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  MAE: %s" % mae)
        print("  MSE: %s" % mse)
        print("  R2: %s" % r2)

        # Log parameters
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)

        # Log metrics
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)

        os.environ['MLFLOW_TRACKING_URI'] = "http://ec2-3-86-151-183.compute-1.amazonaws.com:5000/"
        remote_server_uri = "http://ec2-3-86-151-183.compute-1.amazonaws.com:5000/"
        mlflow.set_tracking_uri(remote_server_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        # Log model
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(model, "model", registered_model_name="ElasticnetWineModel")
            logger.info("Model saved in run %s" % mlflow.active_run().info.run_uuid)
        else:
            mlflow.sklearn.log_model(model, "model")
            print("Model saved in run %s" % mlflow.active_run().info.run_uuid)
