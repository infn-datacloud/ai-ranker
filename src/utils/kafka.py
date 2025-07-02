import json
from logging import Logger
from typing import Any

from kafka import KafkaConsumer, KafkaProducer, TopicPartition
from kafka.errors import NoBrokersAvailable

from src.exceptions import ConfigurationError
from src.settings import InferenceSettings, TrainingSettings


def add_ssl_parameters(
    settings: InferenceSettings | TrainingSettings,
) -> dict[str, Any]:
    """Add SSL configuration parameters for Kafka connection based on provided settings.

    This function reads the SSL password from a file specified in the settings and
    constructs a dictionary of SSL-related keyword arguments required for secure Kafka
    communication.

    Args:
        settings (InferenceSettings | TrainingSettings): The settings object containing
            Kafka SSL configuration paths.
        logger (Logger): Logger instance for logging SSL enablement information.

    Returns:
        dict[str, Any]: A dictionary containing SSL configuration parameters for Kafka.

    Raises:
        ValueError: If the KAFKA_SSL_PASSWORD_PATH is None when SSL is enabled.
        FileNotFoundError: if the KAFKA_SSL_PASSWORD_PATH does not exist.

    """
    if settings.KAFKA_SSL_PASSWORD_PATH is None:
        raise ValueError(
            "KAFKA_SSL_PASSWORD_PATH can't be None when KAFKA_SSL_ENABLE is True"
        )
    with open(settings.KAFKA_SSL_PASSWORD_PATH) as reader:
        ssl_password = reader.read()
    kwargs = {
        "security_protocol": "SSL",
        "ssl_check_hostname": False,
        "ssl_cafile": settings.KAFKA_SSL_CACERT_PATH,
        "ssl_certfile": settings.KAFKA_SSL_CERT_PATH,
        "ssl_keyfile": settings.KAFKA_SSL_KEY_PATH,
        "ssl_password": ssl_password,
    }
    return kwargs


def create_kafka_consumer(
    *,
    kafka_server_url: str,
    topic: str,
    partition: int | None = None,
    offset: int = 0,
    consumer_timeout_ms: int = 0,
    auto_offset_reset: str = "earliest",
    logger: Logger,
) -> KafkaConsumer:
    """Create kafka consumer.

    By default, when starting up, read all messages from beginning.
    It will be a service duty to discard already processed ones.
    """
    if consumer_timeout_ms == 0:
        consumer_timeout_ms = float("inf")

    if partition is None:
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=kafka_server_url,
            auto_offset_reset=auto_offset_reset,
            consumer_timeout_ms=consumer_timeout_ms,
            # enable_auto_commit=False,
            value_deserializer=lambda x: json.loads(
                x.decode("utf-8")
            ),  # Deserialize JSON
        )
        if consumer.bootstrap_connected():
            logger.info("Subscribed to topics: %s", consumer.subscription())

    else:
        consumer = KafkaConsumer(
            bootstrap_servers=kafka_server_url,
            auto_offset_reset=auto_offset_reset,
            consumer_timeout_ms=consumer_timeout_ms,
            # enable_auto_commit=False,
            value_deserializer=lambda x: json.loads(
                x.decode("utf-8")
            ),  # Deserialize JSON
        )
        tp = TopicPartition(topic, partition)
        consumer.assign([tp])
        consumer.seek(tp, offset)
        if consumer.bootstrap_connected():
            logger.info("Assigned topic: %s", consumer.assignment())

    return consumer


def create_kafka_producer(
    *, settings: InferenceSettings, logger: Logger
) -> KafkaProducer:
    """Create and configure a KafkaProducer instance based on the provided settings.

    This function sets up a Kafka producer with JSON value serialization, idempotence,
    and other options as specified in the `settings` object. If SSL is enabled, it loads
    the necessary SSL certificates and password from the provided paths.

    Args:
        settings (InferenceSettings): Configuration object containing Kafka connection
            and security settings.
        logger (Logger): Logger instance for logging errors and information.

    Returns:
        KafkaProducer: Configured Kafka producer instance.

    Raises:
        ConfigurationError: If the Kafka broker is unavailable, required files are
            missing, or configuration is invalid.

    """
    kwargs = {
        "client_id": settings.KAFKA_INFERENCE_CLIENT_NAME,
        "bootstrap_servers": settings.KAFKA_HOSTNAME,
        "value_serializer": lambda x: json.dumps(x, sort_keys=True).encode("utf-8"),
        "max_request_size": settings.KAFKA_MAX_REQUEST_SIZE,
        "acks": "all",
        "enable_idempotence": True,
        "allow_auto_create_topics": settings.KAFKA_ALLOW_AUTO_CREATE_TOPICS,
    }

    try:
        if settings.KAFKA_SSL_ENABLE:
            logger.info("SSL enabled")
            ssl_kwargs = add_ssl_parameters(settings=settings, logger=logger)
            kwargs = {**kwargs, **ssl_kwargs}

        return KafkaProducer(**kwargs)

    except NoBrokersAvailable as e:
        msg = f"Kakfa Broker not found at given url: {settings.KAFKA_HOSTNAME}"
        raise ConfigurationError(msg) from e
    except FileNotFoundError as e:
        msg = f"File '{settings.KAFKA_SSL_PASSWORD_PATH}' not found"
        raise ConfigurationError(msg) from e
    except ValueError as e:
        msg = e.args[0]
        raise ConfigurationError(msg) from e
