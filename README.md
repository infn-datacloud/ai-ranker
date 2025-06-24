# ai-ranker

The **ai-ranker** consists of two main scripts:  
- one for training Machine Learning (ML) models and storing them in an [MLflow](https://mlflow.org) registry;  
- one for performing inference using the trained models retrieved from the registry.

The system relies on an MLflow instance to manage model versions and a [Kafka](https://kafka.apache.org) instance to handle message-based communication for both training and inference. The Kafka instance is not mandatory, as messages can also be read from a local file, although this is intended for debugging purposes only. Currently, only models built with the Scikit-learn library are supported.

## Service logic

The typical usage pattern is as follows:
- The training script is scheduled to run at regular intervals (e.g., via a job scheduler). It consumes messages from the Kafka `training` queue, preprocesses them, and uses them to train two types of ML models: a classification model for predicting the success or failure of a deployment, and a regression model for estimating deployment creation or failure time. The trained models, along with their metadata, are stored in the MLflow registry.
- The inference script listens to the Kafka `inference` queue, reads new messages as they arrive, preprocesses them, loads the latest ML models from the MLflow registry, and uses them to make predictions for each Cloud provider in the message. The predictions are then combined to produce an ordered list of providers based on expected performance.


## Production deployment
### Requirements
You need to have `docker` installed on your system and a running MLflow instance with version 2.17.0. The minimum and recommended resources to use the ai-ranker for training and inference are the following

| Resource               | Minimum (dev/test)                             | Recommended (prod)                          | Notes                                                               |
| ---------------------- | ---------------------------------------------- | ------------------------------------------------ | ------------------------------------------------------------------- |
| **CPU**                | 1 core                                         | 2–4 cores                                        | Scikit-learn does not use GPU but benefits from parallel processing |
| **RAM**                | 2 GB                                           | 4–8 GB                                           | Increase for large datasets or ensemble models                      |
| **Disk (Storage)**     | 1 GB (small models and minimal artifacts)      | 5–10 GB                                          | For storing models, scalers, CSVs, feature importances, etc.        |
| **Network**            | 10 Mbps                                        | 100 Mbps+                                        | Useful if saving/loading artifacts from remote tracking server      |

### Start up the main service
In production mode you should run the application using the dedicated image [indigopaas/ai-ranker](https://hub.docker.com/r/indigopaas/ai-ranker) available on DockerHub.

The application does not requires persistent volumes. It uses environment variables to configure the service: the connection to the MLflow and Kafka instances, the models and features to use, whether to remove outliers, whether to scale the features, and which feature to use for classification and regression.

The command to correctly start up the application for training inside a container using the default environment variables is:

````
docker run indigopaas/ai-ranker python /app/src/main.py --training

````
whereas for inference is:
````
docker run indigopaas/ai-ranker python /app/src/main.py --inference

````
Below is the list of all environment variables that can be passed to the command using the `-e` flag, grouped by category. You can also create a `.env` file with all the variables you want to override.

#### Common Settings

- **LOCAL_MODE**
  - Type: `bool`
  - Description: Whether to perform training using a local dataset
  - Default: `false`

- **LOCAL_DATASET**
  - Type: `str` or `null`
  - Description: Name of the file containing the local dataset
  - Default: `null`

- **LOCAL_DATASET_VERSION**
  - Type: `str`
  - Description: Target message version used to build the dataset
  - Default: `1.1.0`

- **KAFKA_HOSTNAME**
  - Type: `str`
  - Description: Kafka broker address
  - Default: `localhost:9092`

- **KAFKA_TRAINING_TOPIC**
  - Type: `str`
  - Description: Kafka topic for training messages
  - Default: `training`

- **KAFKA_TRAINING_TOPIC_PARTITION**
  - Type: `int` or `null`
  - Description: Kafka partition assigned to this training consumer
  - Default: `null`

- **KAFKA_TRAINING_TOPIC_OFFSET**
  - Type: `int`
  - Description: Offset for reading messages from the training topic
  - Default: `0`

- **KAFKA_TRAINING_TOPIC_TIMEOUT**
  - Type: `int`
  - Description: Milliseconds to wait for a message on the training topic
  - Default: `1000`

- **TEMPLATE_COMPLEX_TYPES**
  - Type: `list`
  - Description: List of complex types used in the template
  - Default: `[]`

---

#### MLFlow Settings

- **MLFLOW_TRACKING_URI**
  - Type: `str` (URL)
  - Description: MLflow tracking URI
  - Default: `http://localhost:5000`

- **MLFLOW_EXPERIMENT_NAME**
  - Type: `str`
  - Description: Name of the MLflow experiment
  - Default: `Default`

- **MLFLOW_HTTP_REQUEST_TIMEOUT**
  - Type: `int`
  - Description: Timeout in seconds for MLflow HTTP requests
  - Default: `20`

- **MLFLOW_HTTP_REQUEST_MAX_RETRIES**
  - Type: `int`
  - Description: Maximum number of retries with exponential backoff for MLflow HTTP requests
  - Default: `5`

- **MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR**
  - Type: `int`
  - Description: Backoff increase factor between MLflow HTTP request retries
  - Default: `2`

- **MLFLOW_HTTP_REQUEST_BACKOFF_JITTER**
  - Type: `float`
  - Description: Jitter added to backoff for MLflow HTTP request retries
  - Default: `1.0`

---

#### Training Settings

- **CLASSIFICATION_MODELS**
  - Type: `dict[str, dict]`
  - Description: JSON string of classifiers and their parameters
  - Default: `{"RandomForestClassifier": {}}`

- **REGRESSION_MODELS**
  - Type: `dict[str, dict]`
  - Description: JSON string of regressors and their parameters
  - Default: `{"RandomForestRegressor": {}}`

- **KFOLDS**
  - Type: `int`
  - Description: Number of folds for KFold cross-validation
  - Default: `5`

- **REMOVE_OUTLIERS**
  - Type: `bool`
  - Description: Whether to remove outliers from the dataset
  - Default: `false`

- **TEST_SIZE**
  - Type: `float`
  - Description: Test set proportion (between 0 and 1)
  - Default: `0.2`

- **Q1_FACTOR**
  - Type: `float`
  - Description: First quantile factor for outlier detection
  - Default: `0.25`

- **Q3_FACTOR**
  - Type: `float`
  - Description: Third quantile factor for outlier detection
  - Default: `0.75`

- **THRESHOLD_FACTOR**
  - Type: `float`
  - Description: Multiplier for outlier threshold
  - Default: `1.5`

- `X_FEATURES`
  - **Description**: List of features to use as X
  - **Type**: list
  - **Default**:
    ```json
    [
      "cpu_diff",
      "ram_diff",
      "storage_diff",
      "instances_diff",
      "floatingips_diff",
      "gpu",
      "test_failure_perc_30d",
      "overbooking_ram",
      "avg_success_time",
      "avg_failure_time",
      "failure_percentage",
      "complexity"
    ]
    ```

- **Y_CLASSIFICATION_FEATURES**
  - Type: `list`
  - Description: List of target feature names for classification
  - Default: `["status"]`

- **Y_REGRESSION_FEATURES**
  - Type: `list`
  - Description: List of target feature names for regression
  - Default: `["deployment_time"]`

- **SCALING_ENABLE**
  - Type: `bool`
  - Description: Whether to scale X features
  - Default: `false`

- **SCALER_FILE**
  - Type: `str`
  - Description: Filename to store the scaler
  - Default: `scaler.pkl`

---

#### Inference Settings

- **CLASSIFICATION_MODEL_NAME**
  - Type: `str`
  - Description: Name of the classification model to load
  - Default: `RandomForestClassifier`

- **CLASSIFICATION_MODEL_VERSION**
  - Type: `str`
  - Description: MLflow model version for classification
  - Default: `latest`

- **CLASSIFICATION_WEIGHT**
  - Type: `float`
  - Description: Weight for classification in the ranking (between 0 and 1)
  - Default: `0.75`

- **REGRESSION_MODEL_NAME**
  - Type: `str`
  - Description: Name of the regression model to load
  - Default: `RandomForestRegressor`

- **REGRESSION_MODEL_VERSION**
  - Type: `str`
  - Description: MLflow model version for regression
  - Default: `latest`

- **LOCAL_IN_MESSAGES**
  - Type: `str` or `null`
  - Description: Path to the local input messages file
  - Default: `null`

- **LOCAL_OUT_MESSAGES**
  - Type: `str` or `null`
  - Description: Path to the local output messages file
  - Default: `null`

- **FILTER**
  - Type: `bool`
  - Description: Whether to filter out results under the threshold
  - Default: `false`

- **THRESHOLD**
  - Type: `float`
  - Description: Minimum score to include a provider in the result
  - Default: `0.7`

- **EXACT_RESOURCES_PRECEDENCE**
  - Type: `bool`
  - Description: Whether to sort providers by how closely they match requested resources
  - Default: `true`

- **KAFKA_INFERENCE_TOPIC**
  - Type: `str`
  - Description: Kafka topic for inference input
  - Default: `inference`

- **KAFKA_INFERENCE_TOPIC_PARTITION**
  - Type: `int` or `null`
  - Description: Kafka partition for inference input
  - Default: `null`

- **KAFKA_INFERENCE_TOPIC_OFFSET**
  - Type: `int`
  - Description: Offset for reading from the inference topic
  - Default: `0`

- **KAFKA_INFERENCE_TOPIC_TIMEOUT**
  - Type: `int`
  - Description: Milliseconds to wait for messages from the inference topic
  - Default: `0`

- **KAFKA_RANKED_PROVIDERS_TOPIC**
  - Type: `str`
  - Description: Kafka topic for ranked output
  - Default: `ranked_providers`

- **KAFKA_RANKED_PROVIDERS_TOPIC_PARTITION**
  - Type: `int` or `null`
  - Description: Kafka partition for ranked output
  - Default: `null`

- **KAFKA_RANKED_PROVIDERS_TOPIC_OFFSET**
  - Type: `int`
  - Description: Offset for reading from the ranked output topic
  - Default: `0`

- **KAFKA_RANKED_PROVIDERS_TOPIC_TIMEOUT**
  - Type: `int`
  - Description: Milliseconds to wait when reading ranked messages
  - Default: `1000`

## Ancillary services
To function correctly, the application requires running instances of both **MLflow** and **Kafka**.

If you don’t already have them running, we recommend deploying them using the official Docker images:
- [ghcr.io/mlflow/mlflow:v2.17.0](https://github.com/mlflow/mlflow/pkgs/container/mlflow/289930635?tag=v2.17.0)
- [confluentinc/cp-kafka](https://hub.docker.com/r/confluentinc/cp-kafka)

As said at the beginning, the Kafka instance is not mandatory, as messages can also be read from a local file, although this is intended for debugging purposes only.

### Job Scheduler
As previously said you can use a container job scheduler to execute the ai-ranker training script contained in the docker image at fixed intervals.
Here, we suggest to use ofelia and we provide an example configuration for a docker-compose.yaml

````
services:
  ai-ranker-training:
    image: indigopaas/ai-ranker
    container_name: ai-ranker-training
    env_file:
      - .env
    restart: unless-stopped
    command: bash -c "while true; do sleep 1; done"
    labels:
      ofelia.enabled: true
      ofelia.job-exec.feed.schedule: "@every 1m"
      ofelia.job-exec.feed.command: "python /app/src/main.py --training"
      ofelia.job-exec.feed.save-folder: /var/log
      ofelia.job-exec.feed.no-overlap: false

  ofelia:
    image: mcuadros/ofelia:latest
    container_name: training-job-scheduler
    depends_on:
      - ai-ranker-training
    restart: unless-stopped
    command: daemon --docker
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - training-logs:/var/log

volumes:
  training-logs:
````

## Developers
### Installation
Clone this repository and move inside the project top folder.
`````
git clone https://github.com/infn-datacloud/ai-ranker
cd ai-ranker
`````

### Setting up environment for local development
Developers can launch the project locally or using containers.
In both cases, developers need a MLflow service and a Kafka service. Both can be started using docker.