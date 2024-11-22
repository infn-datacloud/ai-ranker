import sys
import json
import mlflow
import mlflow.sklearn
from settings import load_mlflow_settings
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
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

def train_model(model_type: str, model_params: dict, experiment_name: str = "Default"):
    """
    Function to train a generic sklearn ML model

    :param model_type: Name of the model to train (e.g. 'RandomForestClassifier').
    :param model_params: Parameters to pass to the model as a dictionary.
    :param experiment_name: Name of the MLFlow experiment
    """
    # Set the MLFlow experiment
    mlflow.set_experiment(experiment_name)
    
    # Get all sklearn models (both classifiers and regressors)
    estimators = dict(all_estimators())
    
    if model_type not in estimators:
        print(f"Error: the model '{model_type}' is not available in scikit-learn.")
        return

    # Load the dataset (here the Iris example)
    file = preprocess_dataset("fullDataset.csv", all)
    X_train, X_test, y_train, y_test = train_test_split(file.iloc[:,:-2], file.iloc[:,-2:-1], test_size=0.2, random_state=42)

    # Load the model dinamically
    ModelClass = estimators[model_type]

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

if __name__ == "__main__":
    settings = load_mlflow_settings()
    print(settings)

    #Set the mlflow server uri
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

    model_params = json.loads(settings.MODELS_PARAMS)

    if len(settings.MODELS) == 1:
        # Train the model chosen by the user
        train_model(settings.MODELS[0], model_params[settings.MODELS[0]], settings.EXPERIMENT_NAME)
    else:
        # Perform KFold cross validation
        kfold_cross_validation(settings.MODELS, model_params, settings.KFOLDS)