import json
from logging import Logger
from tempfile import NamedTemporaryFile

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.utils import all_estimators

import processing
from training import settings
from training.settings import load_airankertraining_settings

singleVM = ["single-vm/single_vm.yaml", "single-vm/single_vm_with_volume.yaml", "single-vm/private-net/single_vm.yaml", "single-vm/private-net/single_vm_with_volume.yaml"]
singleVMComplex = ["single-vm/cloud_storage_service.yaml", "single-vm/elasticsearch_kibana.yaml", "single-vm/iam_voms-aa.yaml"]
k8s = ["kubernetes/k8s_cluster.yaml", "kubernetes/k8s_cluster_with_addons.yaml", "kubernetes/htcondor_k8s.yaml", "kubernetes/private-net/k8s_cluster.yaml", "kubernetes/spark_cluster.yaml"]
docker = ["docker/run_docker.yaml", "docker/docker_compose.yaml", "docker/docker_compose_with_volume.yaml", "docker/run_docker_with_volume.yaml"]
jupyter = ["jupyter/jupyter_vm.yaml", "jupyter/jupyter_matlab.yaml", "jupyter/ml_infn.yaml", "jupyter/cygno_experiment.yaml", "jupyter/private-net/jupyter_vm.yaml "]
all = singleVM + singleVMComplex + k8s + docker + jupyter
simple = singleVM + docker
complex = singleVMComplex + k8s + jupyter

def load_local_dataset(filename: str):
    df = pd.read_csv(f"../dataset/{filename}")
    return df

def set_metadata(df:pd.DataFrame):
    metadata = {
        "start_time": df["timestamp"].max() ,
        "end_time": df["timestamp"].min(),
        "features": settings.FINAL_FEATURES,
        "features_number": len(settings.FINAL_FEATURES),
        "remove outliers": settings.REMOVE_OUTLIERS
    }
    return metadata


def remove_outliers(file: pd.DataFrame):
    # Compute quantiles for all the columns
    Q1 = file.quantile(0.25)
    Q3 = file.quantile(0.75)
    IQR = Q3 - Q1

    # Define limits for the outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Remove outliers in all the columns
    return file[~((file < lower_bound) | (file > upper_bound)).any(axis=1)]

def remove_outliers(X: pd.DataFrame, y: pd.DataFrame):
    # Concat X and y
    combined = pd.concat([X, y], axis=1)

    # Compute quantile on X featutes
    Q1 = combined.quantile(0.25)
    Q3 = combined.quantile(0.75)
    IQR = Q3 - Q1

    # Define limits for the outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Remove outliers both for X and y
    filtered = combined[~((combined < lower_bound) | (combined > upper_bound)).any(axis=1)]

    # Separate X and y
    return filtered.iloc[:, :-1], filtered.iloc[:, -1]

def preprocess_dataset(filename: str, acceptedTemplate: list):
    file = load_local_dataset(filename)
    finalKeys = ["cpu_diff", "ram_diff", "storage_diff", "instances_diff",
                 "floatingips_diff", 'gpu', 'test_failure_perc_30d',
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
    #file["sla_failure_percentage"] = file["sla_failure_percentage"].replace(mapSla).astype(float)
    file["test_failure_perc_30d"] = file["sla_failure_percentage"].replace(mapSla).astype(float)
    file = file[file["avg_deployment_time"].notna()]
    #file = file[file['difference'] < 10000]

    file["cpu_diff"] = (file["quota_cores"] - file["vcpus"]) - file["requested_cpu"]
    file["ram_diff"] = (file["quota_ram"] - file["ram"]) - file["requested_ram"]
    file["storage_diff"] = (file["quota_archived"] - file["archived"]) - file["requested_storage"]
    file["instances_diff"] = (file["quota_instances"] - file["instances"]) - file["requested_nodes"]
    file["volumes_diff"] = (file["quota_volumes"] - file["volumes"]) - file["requested_volumes"]
    file["floatingips_diff"] = (file["quota_floatingips"] - file["floatingips"])
    metadata = {
        "start_time": file["creation_time"].iloc[0],
        "end_time": file["creation_time"].iloc[-1],
        "features": file[finalKeys].columns.to_list(),
        "features_number": len(file[finalKeys].columns),
        "remove outliers": settings.REMOVE_OUTLIERS
    }
    return file[finalKeys], metadata

def kfold_cross_validation(X_train, X_test, y_train, y_test, metadata, models_params, n_splits=5, scoring="roc_auc"):
    X = X_train
    y = y_train

    # Initialize KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Dictionary to store scores
    model_scores = {}
    mean_scores = {}
    # Get all estimators
    all_models = dict(all_estimators())

    # Initialize RobustScaler
    scaler = RobustScaler()

    for model_name, params in models_params.items():
        # Fetch the model class
        ModelClass = all_models.get(model_name)
        if ModelClass is None:
            raise ValueError(f"Model {model_name} not found in sklearn estimators.")

        # Instantiate the model with parameters
        model = ModelClass(**params)

        # Scale the features
        X_scaled = scaler.fit_transform(X)

        # Perform cross-validation
        scores = cross_val_score(model, X_scaled, y.values.ravel(), cv=kf, scoring=scoring)

        # Store the scores in the dictionary
        model_scores[model_name] = scores
        mean_scores[model_name] = np.mean(scores)
        print(f"Model: {model_name}, Mean {scoring}: {np.mean(scores):.4f}, Std: {np.std(scores):.4f}")
    best_model_name = max(mean_scores, key=mean_scores.get)
    print(f"Model selected for training: {best_model_name}")
    if issubclass(all_models.get(best_model_name), ClassifierMixin):
        train_model_classification(X_train, X_test, y_train, y_test, metadata, best_model_name, models_params[best_model_name])
    elif issubclass(all_models.get(best_model_name), RegressorMixin):
        train_model_regression(X_train, X_test, y_train, y_test, metadata, best_model_name, models_params[best_model_name])

    return model_scores

def get_feature_importance(model, columns, X_train_scaled):
    # Case 1: Models with attribute `feature_importances_`
    if hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': columns,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)
        return feature_importance_df

    # Case 2: Models like Lasso (coefficient)
    elif hasattr(model, 'coef_'):
        coef = model.coef_
        feature_importance_df = pd.DataFrame({
            'Feature': columns,
            'Coefficient': coef
        }).sort_values(by='Coefficient', ascending=False)
        return feature_importance_df

    # Case 3: Models without `feature_importances_` or `coef_`, use SHAP
    else:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_train_scaled)
            shap.summary_plot(shap_values, X_train_scaled)
            # Gives mean values of feayure importance
            shap_importance = np.mean(np.abs(shap_values), axis=0)
            feature_importance_df = pd.DataFrame({
                'Feature': columns,
                'Importance': shap_importance
            }).sort_values(by='Importance', ascending=False)
            return feature_importance_df
        except Exception as e:
            print(f"Errore nell'uso di SHAP: {e}")
            return None


def train_model_classification(X_train, X_test, y_train, y_test, metadata, model_type: str, model_params: dict):
    """
    Function to train a generic sklearn ML model

    :param model_type: Name of the model to train (e.g. 'RandomForestClassifier').
    :param model_params: Parameters to pass to the model as a dictionary.
    :param experiment_name: Name of the MLFlow experiment
    """

    # Load the model dinamically
    ModelClass = dict(all_estimators())[model_type]

    # Create the model with the requested parameters
    try:
        model = ModelClass(**model_params)
    except TypeError as e:
        print(f"Error in the creation of the model: {e}")
        return

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the model
    model.fit(X_train_scaled, y_train.values.ravel())

    # Get feature importance
    feature_importance_df = get_feature_importance(model, X_train.columns, X_train_scaled)

    if feature_importance_df is not None:
        print(feature_importance_df)

    # Do predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    # Compute the accuracy if the model is a classifier
    if isinstance(model, ClassifierMixin):
        metrics = {
            "Accuracy train": accuracy_score(y_train, y_pred_train),
            "auc train": roc_auc_score(y_train, model.predict_proba(X_train_scaled)[:,1]),
            "F1 train": f1_score(y_train, y_pred_train, average='binary'),
            "Precision train": precision_score(y_train, y_pred_train, average='binary'),
            "Recall train": recall_score(y_train, y_pred_train, average='binary'),
            "Accuracy test": accuracy_score(y_test, y_pred_test),
            "auc test": roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:,1]),
            "F1 test": f1_score(y_test, y_pred_test, average='binary'),
            "Precision test": precision_score(y_test, y_pred_test, average='binary'),
            "Recall test": recall_score(y_test, y_pred_test, average='binary')
        }
        print(confusion_matrix(y_test, y_pred_test))
        print(classification_report(y_test, y_pred_test))
        log_on_mlflow(model_params, model_type, model, metrics, metadata, feature_importance_df)


def train_model_regression(X_train, X_test, y_train, y_test, metadata, model_type: str, model_params: dict):

    # Load the model dinamically
    ModelClass = dict(all_estimators())[model_type]

    # Create the model with the requested parameters
    try:
        model = ModelClass(**model_params)
    except TypeError as e:
        print(f"Error in the creation of the model: {e}")
        return

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the model
    model.fit(X_train_scaled, y_train.values.ravel())

    # Get feature importance
    feature_importance_df = get_feature_importance(model, X_train.columns, X_train_scaled)

    if feature_importance_df is not None:
        print(feature_importance_df)

    # Do predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    if isinstance(model, RegressorMixin):
        # Compute metrics
        metrics = {
            "MSE train": mean_squared_error(y_train, y_train_pred),
            "MSE test": mean_squared_error(y_test, y_test_pred),
            "RMSE train": np.sqrt(mean_squared_error(y_train, y_train_pred)),
            "RMSE test": np.sqrt(mean_squared_error(y_test, y_test_pred)),
            "MAE train": mean_absolute_error(y_train, y_train_pred),
            "MAE test": mean_absolute_error(y_test, y_test_pred),
            "R2 train": r2_score(y_train, y_train_pred),
            "R2 test": r2_score(y_test, y_test_pred),
        }

        log_on_mlflow(model_params, model_type, model, metrics, metadata, feature_importance_df)

def log_on_mlflow(model_params: dict, model_type: str, model: any, metrics: dict, metadata: dict, feature_importance_df: pd.DataFrame):
    # Logging on MLflow
    with mlflow.start_run():
        # Log the parameters
        mlflow.log_params(model_params)

        # Print the metrics
        for metric, value in metrics.items():
            print(f"{metric}: {value}")

        # Log the metrics
        for metric, value in metrics.items():
            mlflow.log_metric(metric, value)

        # Log the sklearn model and register
        mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=model_type,
        registered_model_name=model_type,
        metadata=metadata
        )
        for key, value in metadata.items():
            mlflow.set_tag(key, value)

        with NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            feature_importance_df.to_csv(temp_file.name, index=False)
            mlflow.log_artifact(temp_file.name, artifact_path="feature_importances")
    print(f"Model {model_type} successfully logged on MLflow.")

def setup_mlflow():
    # Load the settings
    settings = load_airankertraining_settings()

    # Set the mlflow server uri
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

    # Set the MLFlow experiment
    mlflow.set_experiment(settings.EXPERIMENT_NAME)

    # Get all sklearn models (both classifiers and regressors)
    estimators = dict(all_estimators())

    for model in settings.CLASSIFICATION_MODELS + settings.REGRESSION_MODELS:
        if model not in estimators:
            print(f"Error: the model '{model}' is not available in scikit-learn.")
            return

    return settings

def run(logger: Logger) -> None:
    settings = setup_mlflow()
    classification_model_params = json.loads(settings.CLASSIFICATION_MODELS_PARAMS)
    regression_model_params = json.loads(settings.REGRESSION_MODELS_PARAMS)
    # Load the dataset (here the Iris example)
    if settings.LOCAL_MODE:
        file = load_local_dataset(settings.LOCAL_DATASET)
        #file, metadata = preprocess_dataset(settings.LOCAL_DATASET, all)
    else: 
        file = processing.load_dataset_from_kafka(kafka_server_url="localhost:9092", topic="training", partition=0, offset=765)
    metadata = set_metadata(file)
    df = processing.preprocessing(file, settings.TEMPLATE_COMPLEX_TYPES)
    df = processing.filter_df(df, settings.FINAL_FEATURES)
    print(df) 

    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,:-2], df.iloc[:,-2:-1], test_size=0.2, random_state=42)
    if settings.REMOVE_OUTLIERS:
        X_train_cleaned, y_train_cleaned = remove_outliers(X_train, y_train)
    else:
        X_train_cleaned, y_train_cleaned = X_train, y_train
    if len(settings.CLASSIFICATION_MODELS) == 1:
        # Train the model chosen by the user
        train_model_classification(X_train_cleaned, X_test, y_train_cleaned, y_test, metadata, settings.CLASSIFICATION_MODELS[0], classification_model_params[settings.CLASSIFICATION_MODELS[0]])
    else:
        # Perform KFold cross validation
        kfold_cross_validation(X_train_cleaned, X_test, y_train_cleaned, y_test, metadata, classification_model_params, settings.KFOLDS, scoring="roc_auc")

    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,:-2], df.iloc[:,-1:], test_size=0.2, random_state=42)
    if settings.REMOVE_OUTLIERS:
        X_train_cleaned, y_train_cleaned = remove_outliers(X_train, y_train)
    else:
        X_train_cleaned, y_train_cleaned = X_train, y_train
    if len(settings.REGRESSION_MODELS) == 1:
        # Train the model chosen by the user
        train_model_regression(X_train_cleaned, X_test, y_train_cleaned, y_test, metadata, settings.REGRESSION_MODELS[0], regression_model_params[settings.REGRESSION_MODELS[0]])
    else:
        #Perform KFold cross validation
        kfold_cross_validation(X_train_cleaned, X_test, y_train_cleaned, y_test, metadata, regression_model_params, settings.KFOLDS, scoring="r2")
