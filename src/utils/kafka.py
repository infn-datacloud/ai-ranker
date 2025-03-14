import json
from logging import Logger

from kafka import KafkaConsumer, KafkaProducer  # , TopicPartition
from kafka.errors import NoBrokersAvailable


def create_kafka_consumer(
    *,
    kafka_server_url: str,
    topic: str,
    logger: Logger,
    partition: int = 0,
    offset: int = 0,
    consumer_timeout_ms: int = 0,
) -> KafkaConsumer:
    """Create kafka consumer."""
    if consumer_timeout_ms == 0:
        consumer_timeout_ms = float("inf")
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=kafka_server_url,
        auto_offset_reset="earliest",
        # enable_auto_commit=False,
        value_deserializer=lambda x: json.loads(
            x.decode("utf-8")
        ),  # deserializza il JSON
        consumer_timeout_ms=consumer_timeout_ms,
    )

    # TODO: Manage automatic partitioning
    # tp = TopicPartition(topic, partition)
    # consumer.assign([tp])
    # consumer.seek(tp, offset)
    if consumer.bootstrap_connected():
        logger.info("Subscribed to topics: %s", consumer.subscription())
    # TODO: Manage disconnection and reconnections
    # attempt = 0
    # while True:
    #     if not consumer.bootstrap_connected():
    #         if attempt < settings.KAFKA_RECONNECT_MAX_RETRIES:
    #             logger.warning(
    #                 "Can't connect to topic '%s' on server '%s'. Waiting for %ss",
    #                 settings.KAFKA_TRAINING_TOPIC,
    #                 settings.KAFKA_HOSTNAME,
    #                 settings.KAFKA_RECONNECT_PERIOD,
    #             )
    #             attempt += 1
    #         else:
    #             logger.error(
    #                 "connect to topic '%s' on server '%s'. Max attempt reached (%s)",
    #                 settings.KAFKA_TRAINING_TOPIC,
    #                 settings.KAFKA_HOSTNAME,
    #                 settings.KAFKA_RECONNECT_MAX_RETRIES,
    #             )
    #         exit(1)
    #     else:
    #         attempt = 0
    return consumer


def create_kafka_producer(*, kafka_server_url: str, logger: Logger) -> KafkaProducer:
    """Create a kafka producer"""
    try:
        producer = KafkaProducer(
            bootstrap_servers=kafka_server_url,
            client_id="inference",
            value_serializer=lambda x: json.dumps(x).encode("utf-8"),
        )
    except NoBrokersAvailable:
        logger.error("Kakfa Broker not found at given url: %s", kafka_server_url)
        exit(1)
    return producer
