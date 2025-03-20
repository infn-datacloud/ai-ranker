import json
from logging import Logger

from kafka import KafkaConsumer, KafkaProducer, TopicPartition
from kafka.errors import NoBrokersAvailable


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


def create_kafka_producer(*, kafka_server_url: str, logger: Logger) -> KafkaProducer:
    """Create a kafka producer"""
    try:
        producer = KafkaProducer(
            bootstrap_servers=kafka_server_url,
            value_serializer=lambda x: json.dumps(x).encode("utf-8"),  # Serialize JSON
        )
    except NoBrokersAvailable:
        logger.error("Kakfa Broker not found at given url: %s", kafka_server_url)
        exit(1)
    return producer
