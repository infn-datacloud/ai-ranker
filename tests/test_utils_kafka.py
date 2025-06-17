from unittest.mock import MagicMock, patch

import pytest
from kafka.errors import NoBrokersAvailable

from src.utils.kafka import create_kafka_consumer, create_kafka_producer


@pytest.fixture
def mock_logger():
    return MagicMock()


@patch("src.utils.kafka.KafkaConsumer")
def test_create_kafka_consumer_default(mock_kafka_consumer, mock_logger):
    consumer_instance = MagicMock()
    consumer_instance.bootstrap_connected.return_value = True
    mock_kafka_consumer.return_value = consumer_instance

    consumer = create_kafka_consumer(
        kafka_server_url="localhost:9092",
        topic="test-topic",
        logger=mock_logger,
    )

    mock_kafka_consumer.assert_called_once()
    consumer_instance.bootstrap_connected.assert_called_once()
    mock_logger.info.assert_called_once()
    assert consumer == consumer_instance


@patch("src.utils.kafka.KafkaConsumer")
@patch("src.utils.kafka.TopicPartition")
def test_create_kafka_consumer_with_partition(mock_topic_partition, mock_kafka_consumer, mock_logger):
    consumer_instance = MagicMock()
    consumer_instance.bootstrap_connected.return_value = True
    mock_kafka_consumer.return_value = consumer_instance

    tp_instance = MagicMock()
    mock_topic_partition.return_value = tp_instance

    consumer = create_kafka_consumer(
        kafka_server_url="localhost:9092",
        topic="test-topic",
        partition=0,
        offset=10,
        logger=mock_logger,
    )

    mock_topic_partition.assert_called_once_with("test-topic", 0)
    consumer_instance.assign.assert_called_once_with([tp_instance])
    consumer_instance.seek.assert_called_once_with(tp_instance, 10)
    mock_logger.info.assert_called_once()
    assert consumer == consumer_instance


@patch("src.utils.kafka.KafkaConsumer")
def test_create_kafka_consumer_timeout_inf(mock_kafka_consumer, mock_logger):
    consumer_instance = MagicMock()
    consumer_instance.bootstrap_connected.return_value = False
    mock_kafka_consumer.return_value = consumer_instance

    consumer = create_kafka_consumer(
        kafka_server_url="localhost:9092",
        topic="topic",
        consumer_timeout_ms=0,
        logger=mock_logger,
    )

    mock_kafka_consumer.assert_called_once()
    mock_logger.info.assert_not_called()
    assert consumer == consumer_instance


@patch("src.utils.kafka.KafkaProducer")
def test_create_kafka_producer_success(mock_kafka_producer, mock_logger):
    producer_instance = MagicMock()
    mock_kafka_producer.return_value = producer_instance

    producer = create_kafka_producer(
        kafka_server_url="localhost:9092",
        logger=mock_logger,
    )

    mock_kafka_producer.assert_called_once()
    assert producer == producer_instance


@patch("src.utils.kafka.KafkaProducer", side_effect=NoBrokersAvailable())
def test_create_kafka_producer_no_broker(mock_kafka_producer, mock_logger):
    with pytest.raises(SystemExit):
        create_kafka_producer(
            kafka_server_url="localhost:9092",
            logger=mock_logger,
        )

    mock_logger.error.assert_called_once_with(
        "Kakfa Broker not found at given url: %s", "localhost:9092"
    )

@patch("src.utils.kafka.KafkaConsumer")
def test_create_kafka_consumer_with_nonzero_timeout(mock_kafka_consumer, mock_logger):
    consumer_instance = MagicMock()
    consumer_instance.bootstrap_connected.return_value = False
    mock_kafka_consumer.return_value = consumer_instance

    consumer = create_kafka_consumer(
        kafka_server_url="localhost:9092",
        topic="topic",
        consumer_timeout_ms=5000,
        logger=mock_logger,
    )

    mock_kafka_consumer.assert_called_once()
    mock_logger.info.assert_not_called()
    assert consumer == consumer_instance

@patch("src.utils.kafka.KafkaConsumer")
def test_create_kafka_consumer_with_nonzero_timeout_other(mock_kafka_consumer, mock_logger):
    consumer_instance = MagicMock()
    consumer_instance.bootstrap_connected.return_value = True
    mock_kafka_consumer.return_value = consumer_instance

    consumer = create_kafka_consumer(
        kafka_server_url="localhost:9092",
        topic="topic",
        consumer_timeout_ms=5000,
        logger=mock_logger,
    )

    mock_kafka_consumer.assert_called_once()
    consumer_instance.bootstrap_connected.assert_called_once()
    mock_logger.info.assert_called_once()
    assert consumer == consumer_instance
