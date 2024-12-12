import sys
import json
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from settings import load_mlflow_settings
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, roc_auc_score, r2_score
from sklearn.utils import all_estimators
from sklearn.base import ClassifierMixin, RegressorMixin
from mlflow.models import infer_signature
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from kafka import KafkaConsumer
import random
import string

singleVM = ["single-vm/single_vm.yaml", "single-vm/single_vm_with_volume.yaml", "single-vm/private-net/single_vm.yaml", "single-vm/private-net/single_vm_with_volume.yaml"]
singleVMComplex = ["single-vm/cloud_storage_service.yaml", "single-vm/elasticsearch_kibana.yaml", "single-vm/iam_voms-aa.yaml"]
k8s = ["kubernetes/k8s_cluster.yaml", "kubernetes/k8s_cluster_with_addons.yaml", "kubernetes/htcondor_k8s.yaml", "kubernetes/private-net/k8s_cluster.yaml", "kubernetes/spark_cluster.yaml"]
docker = ["docker/run_docker.yaml", "docker/docker_compose.yaml", "docker/docker_compose_with_volume.yaml", "docker/run_docker_with_volume.yaml"]
jupyter = ["jupyter/jupyter_vm.yaml", "jupyter/jupyter_matlab.yaml", "jupyter/ml_infn.yaml", "jupyter/cygno_experiment.yaml", "jupyter/private-net/jupyter_vm.yaml "]
all = singleVM + singleVMComplex + k8s + docker + jupyter
simple = singleVM + docker
complex = singleVMComplex + k8s + jupyter

def load_local_dataset(filename: str):
    df = pd.read_csv(f"/home/ubuntu/dataset/{filename}")
    return df

def load_kafka_dataset():
    group_id = ''.join(random.choices(string.ascii_uppercase +
                                  string.ascii_lowercase +
                                  string.digits, k=64))

    consumer = KafkaConsumer(
        'training-ai-ranker',
        bootstrap_servers=['192.168.21.96:9092'],
        group_id=f'rally-group-{group_id}',
        auto_offset_reset='earliest', 
        enable_auto_commit=True,
        value_deserializer=lambda x: json.loads(x),
        consumer_timeout_ms=500
    )
    l_data = [message.value for message in consumer]
    df = pd.DataFrame(l_data)
    return df

def preprocess_dataset(filename: str, acceptedTemplate: list):
    file = load_local_dataset(filename)
    finalKeys = ["cpu_diff", "ram_diff", "storage_diff", "instances_diff",
                 "floatingips_diff", 'gpu', 'sla_failure_percentage',
                 'overbooking_ram', 'avg_deployment_time',
                 'failure_percentage', 'complexity', "status", "difference"]
    file['complexity'] = file['selected_template'].isin(complex).astype(int)
    file.loc[file['selected_template'].isin(simple), 'complexity'] = 0
    if acceptedTemplate != all:
        file = file[file["selected_template"].isin(acceptedTemplate)]
    mapStatus = {"CREATE_COMPLETE": 0, "CREATE_FAILED": 1}
    mapTrueFalse = {True: 1, False: 0, pd.NA: 0}
    mapSla = {"TROPPO PRESTO!!": 0, "No matching entries": 0}
    file["status"] = file["status"].replace(mapStatus).astype(int)
    file["gpu"] = file["gpu"].replace(mapTrueFalse).astype(int)
    file["sla_failure_percentage"] = file["sla_failure_percentage"].replace(mapSla).astype(float)
    file = file[file["avg_deployment_time"].notna()]
    file = file[file['difference'] < 10000]

    file["cpu_diff"] = (file["quota_cores"] - file["vcpus"]) - file["requested_cpu"]
    file["ram_diff"] = (file["quota_ram"] - file["ram"]) - file["requested_ram"]
    file["storage_diff"] = (file["quota_archived"] - file["archived"]) - file["requested_storage"]
    file["instances_diff"] = (file["quota_instances"] - file["instances"]) - file["requested_nodes"]
    file["volumes_diff"] = (file["quota_volumes"] - file["volumes"]) - file["requested_volumes"]
    file["floatingips_diff"] = (file["quota_floatingips"] - file["floatingips"])
    return file[finalKeys]

def kfold_cross_validation(models, model_params, kfolds):
    return

def train_model_classification(model_type: str, model_params: dict, file):
    """
    Function to train a generic sklearn ML model

    :param model_type: Name of the model to train (e.g. 'RandomForestClassifier').
    :param model_params: Parameters to pass to the model as a dictionary.
    :param experiment_name: Name of the MLFlow experiment
    """

    X_train, X_test, y_train, y_test = train_test_split(file.iloc[:,:-2], file.iloc[:,-2:-1], test_size=0.2, random_state=42)
    # Load the model dinamically
    ModelClass = dict(all_estimators())[model_type]

    # Create the model with the requested parameters
    try:
        model = ModelClass(**model_params)
    except TypeError as e:
        print(f"Error in the creation of the model: {e}")
        return

    # Train the model
    model.fit(X_train, y_train)

    # Do predictions
    y_pred = model.predict(X_test)

    # Compute the accuracy if the model is a classifier
    if isinstance(model, ClassifierMixin):
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(file["status"], model.predict_proba(file[file.columns[:-2]].values)[:,1])
        print(f"Accuracy of the model {model_type}: {accuracy}")
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        print(f'AUC: {auc}')
    else:
        print(f"Model {model_type} successfully trained.")

    #Logging on MLflow
    with mlflow.start_run():
        # Log the parameters
        mlflow.log_params(model_params)
        
        # Log the model
        mlflow.sklearn.log_model(model, model_type)
        
        # Log accuracy metric for classifier
        if isinstance(model, ClassifierMixin):
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("auc", accuracy)
        
        #Log the sklearn model and register
        mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=model_type,
        registered_model_name=model_type,
    )

    print(f"Model {model_type} successfully logged on MLflow.")

def train_model_regression(model_type: str, model_params: dict, file):

    X_train, X_test, y_train, y_test = train_test_split(file.iloc[:,:-2], file.iloc[:,-1:], test_size=0.2, random_state=42)
    # Load the model dinamically
    ModelClass = dict(all_estimators())[model_type]

    # Create the model with the requested parameters
    try:
        model = ModelClass(**model_params)
    except TypeError as e:
        print(f"Error in the creation of the model: {e}")
        return

    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the model
    model.fit(X_train_scaled, y_train)

    # Do predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # Compute the metrics
    metrics = {
        "MSE_train": mean_squared_error(y_train, y_train_pred),
        "MSE_test": mean_squared_error(y_test, y_test_pred),
        "RMSE_train": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "RMSE_test": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        "MAE_train": mean_absolute_error(y_train, y_train_pred),
        "MAE_test": mean_absolute_error(y_test, y_test_pred),
        "R2_train": r2_score(y_train, y_train_pred),
        "R2_test": r2_score(y_test, y_test_pred),
    }
    # Log the metrics
    for metric, value in metrics.items():
        mlflow.log_metric(metric, value)

    # Log parameters
    mlflow.log_param("model", "RandomForestRegressor")
    mlflow.log_param("scaling", "MinMaxScaler (-1, 1)")
    mlflow.log_param("test_size", 0.2)

    # Log del modello
    mlflow.sklearn.log_model(model, "model")
    print("Modello logged on MLflow.")

    # Print the metrics
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

def setup_mlflow():
    settings = load_mlflow_settings()
    print(settings)
    #Set the mlflow server uri
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

    # Set the MLFlow experiment
    mlflow.set_experiment(settings.EXPERIMENT_NAME)

    # Get all sklearn models (both classifiers and regressors)
    estimators = dict(all_estimators())

    for model in settings.CLASSIFICATION_MODELS + settings.REGRESSION_MODELS:
        if model not in estimators:
            print(f"Error: the model '{model}' is not available in scikit-learn.")
            return
        else:
            print(model)

    return settings

if __name__ == "__main__":

    settings = setup_mlflow()
    classification_model_params = json.loads(settings.CLASSIFICATION_MODELS_PARAMS)
    regression_model_params = json.loads(settings.REGRESSION_MODELS_PARAMS)

    # Load the dataset (here the Iris example)
    file = preprocess_dataset("fullDataset.csv", all)

    if len(settings.CLASSIFICATION_MODELS) == 1:
        # Train the model chosen by the user
        train_model_classification(settings.CLASSIFICATION_MODELS[0], classification_model_params[settings.CLASSIFICATION_MODELS[0]], file)
    else:
        # Perform KFold cross validation
        kfold_cross_validation(settings.CLASSIFICATION_MODELS, classification_model_params, settings.KFOLDS)

    if len(settings.REGRESSION_MODELS) == 1:
        # Train the model chosen by the user
        train_model_regression(settings.REGRESSION_MODELS[0], regression_model_params[settings.REGRESSION_MODELS[0]], file)
    else:
        # Perform KFold cross validation
        kfold_cross_validation(settings.REGRESSION_MODELS, regression_model_params, settings.KFOLDS)
