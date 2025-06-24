# ai-ranker

The AI-Ranker is an application that allows to train Machine Learning (ML) models, save the training results and models to an MLflow registry, and perform inference. It uses an [MLflow](https://mlflow.org) instance as the model registry and a [Kafka](https://kafka.apache.org) instance to read messages for training and inference. The Kafka instance is not mandatory, as messages can also be read from a local file, although this is intended for debugging purposes only.

## Production deployment
### Requirements
You need to have `docker` installed on your system and a running a MLflow instance. 