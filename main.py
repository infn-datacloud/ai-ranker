import sys
import json
import mlflow
import mlflow.sklearn
from settings import load_mlflow_settings
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
from sklearn.base import ClassifierMixin, RegressorMixin
from mlflow.models import infer_signature
from sklearn.metrics import classification_report, confusion_matrix

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
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
        print(f"Accuracy of the model {model_type}: {accuracy}")
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
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

    # Train the model
    train_model(settings.MODEL_TYPE, settings.MODEL_PARAMS, settings.EXPERIMENT_NAME)
