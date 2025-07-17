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
        ValueError: If the KAFKA_SSL_PASSWORD is None when SSL is enabled.

    """
    if settings.KAFKA_SSL_PASSWORD is None:
        raise ValueError(
            "KAFKA_SSL_PASSWORD can't be None when KAFKA_SSL_ENABLE is True"
        )
    kwargs = {
        "security_protocol": "SSL",
        "ssl_check_hostname": False,
        "ssl_cafile": settings.KAFKA_SSL_CACERT_PATH,
        "ssl_certfile": settings.KAFKA_SSL_CERT_PATH,
        "ssl_keyfile": settings.KAFKA_SSL_KEY_PATH,
        "ssl_password": settings.KAFKA_SSL_PASSWORD,
    }
    return kwargs


def create_kafka_consumer(
    *,
    settings: InferenceSettings | TrainingSettings,
    topic: str,
    client_id: str,
    partition: int | None = None,
    offset: int = 0,
    consumer_timeout_ms: int = 0,
    auto_offset_reset: str = "earliest",
    logger: Logger,
) -> KafkaConsumer:
    """Create a Kafka consumer instance with optional SSL and partition configurations.

    By default, the consumer reads all messages from the beginning of the topic.
    SSL parameters are added if enabled in the settings. If a partition is specified,
    the consumer is assigned to that partition and offset.

    Args:
        settings (InferenceSettings | TrainingSettings): Application settings containing
            Kafka configuration.
        topic (str): The Kafka topic to consume from.
        client_id (str): Unique identifier for the Kafka client.
        partition (int | None, optional): Specific partition to consume from. If None,
            all partitions are used. Defaults to None.
        offset (int, optional): Offset to start consuming from if a partition is
            specified. Defaults to 0.
        consumer_timeout_ms (int, optional): Timeout in milliseconds for the consumer to
            wait for messages. Defaults to 0 (waits indefinitely).
        auto_offset_reset (str, optional): Policy for resetting offsets ('earliest',
            'latest', etc.). Defaults to "earliest".
        logger (Logger): Logger instance for logging events.

    Returns:
        KafkaConsumer: Configured Kafka consumer instance.

    Raises:
        ConfigurationError: If Kafka broker is not available, SSL file is missing, or
            configuration is invalid.

    """
    if consumer_timeout_ms == 0:
        consumer_timeout_ms = float("inf")

    kwargs = {
        "client_id": client_id,
        "bootstrap_servers": settings.KAFKA_HOSTNAME,
        "value_deserializer": lambda x: json.loads(x.decode("utf-8")),
        "fetch_max_bytes": settings.KAFKA_MAX_REQUEST_SIZE,
        "consumer_timeout_ms": consumer_timeout_ms,
        "auto_offset_reset": auto_offset_reset,
        "enable_auto_commit": True,
        "group_id": None,
        # 'group_instance_id': None,
        "max_poll_records": 1,
    }

    try:
        if settings.KAFKA_SSL_ENABLE:
            logger.info("SSL enabled")
            ssl_kwargs = add_ssl_parameters(settings=settings)
            kwargs = {**kwargs, **ssl_kwargs}

        if partition is None:
            logger.info("No partition defined")
            consumer = KafkaConsumer(topic, **kwargs)
        else:
            logger.info(
                "Using partition '%d' and offset '%d' of topic '%s'",
                partition,
                offset,
                topic,
            )
            consumer = KafkaConsumer(**kwargs)
            tp = TopicPartition(topic, partition)
            consumer.assign([tp])
            consumer.seek(tp, offset)

        if consumer.bootstrap_connected():
            logger.info("Assigned topic: %s", consumer.assignment())

        return consumer
    except NoBrokersAvailable as e:
        msg = f"Kakfa Broker not found at given url: {settings.KAFKA_HOSTNAME}"
        raise ConfigurationError(msg) from e
    except ValueError as e:
        msg = e.args[0]
        raise ConfigurationError(msg) from e


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
            ssl_kwargs = add_ssl_parameters(settings=settings)
            kwargs = {**kwargs, **ssl_kwargs}

        return KafkaProducer(**kwargs)

    except NoBrokersAvailable as e:
        msg = f"Kakfa Broker not found at given url: {settings.KAFKA_HOSTNAME}"
        raise ConfigurationError(msg) from e
    except ValueError as e:
        msg = e.args[0]
        raise ConfigurationError(msg) from e
