from unittest.mock import MagicMock, patch

import pytest
from kafka.errors import NoBrokersAvailable

from src.exceptions import ConfigurationError
from src.utils.kafka import create_kafka_consumer, create_kafka_producer


@patch("src.utils.kafka.KafkaConsumer")
def test_create_kafka_consumer_default(mock_kafka_consumer):
    logger = MagicMock()
    mock_settings = MagicMock()
    consumer_instance = MagicMock()
    consumer_instance.bootstrap_connected.return_value = True
    mock_kafka_consumer.return_value = consumer_instance

    consumer = create_kafka_consumer(
        settings=mock_settings,
        client_id="test-client",
        topic="test-topic",
        logger=logger,
    )

    mock_kafka_consumer.assert_called_once()
    consumer_instance.bootstrap_connected.assert_called_once()
    assert logger.info.call_count == 3
    assert consumer == consumer_instance


@patch("src.utils.kafka.KafkaConsumer")
@patch("src.utils.kafka.TopicPartition")
def test_create_kafka_consumer_with_partition(
    mock_topic_partition, mock_kafka_consumer
):
    logger = MagicMock()
    mock_settings = MagicMock()
    consumer_instance = MagicMock()
    consumer_instance.bootstrap_connected.return_value = True
    mock_kafka_consumer.return_value = consumer_instance

    tp_instance = MagicMock()
    mock_topic_partition.return_value = tp_instance

    consumer = create_kafka_consumer(
        settings=mock_settings,
        client_id="test-client",
        topic="test-topic",
        partition=0,
        offset=10,
        logger=logger,
    )

    mock_topic_partition.assert_called_once_with("test-topic", 0)
    consumer_instance.assign.assert_called_once_with([tp_instance])
    consumer_instance.seek.assert_called_once_with(tp_instance, 10)
    assert logger.info.call_count == 3
    assert consumer == consumer_instance


@patch("src.utils.kafka.KafkaConsumer")
def test_create_kafka_consumer_timeout_inf(mock_kafka_consumer):
    logger = MagicMock()
    mock_settings = MagicMock()
    consumer_instance = MagicMock()
    consumer_instance.bootstrap_connected.return_value = False
    mock_kafka_consumer.return_value = consumer_instance

    consumer = create_kafka_consumer(
        settings=mock_settings,
        client_id="test-client",
        topic="topic",
        consumer_timeout_ms=0,
        logger=logger,
    )

    mock_kafka_consumer.assert_called_once()
    assert logger.info.call_count == 2
    assert consumer == consumer_instance


@patch("src.utils.kafka.KafkaConsumer")
def test_create_kafka_consumer_with_nonzero_timeout(mock_kafka_consumer):
    logger = MagicMock()
    mock_settings = MagicMock()
    consumer_instance = MagicMock()
    consumer_instance.bootstrap_connected.return_value = False
    mock_kafka_consumer.return_value = consumer_instance

    consumer = create_kafka_consumer(
        settings=mock_settings,
        client_id="test-client",
        topic="topic",
        consumer_timeout_ms=5000,
        logger=logger,
    )

    mock_kafka_consumer.assert_called_once()
    assert logger.info.call_count == 2
    assert consumer == consumer_instance


@patch("src.utils.kafka.KafkaConsumer")
@patch("src.utils.kafka.TopicPartition")
def test_create_kafka_consumer_with_partition_and_bootstrap_false(
    mock_topic_partition, mock_kafka_consumer
):
    logger = MagicMock()
    mock_settings = MagicMock()
    consumer_instance = MagicMock()
    consumer_instance.bootstrap_connected.return_value = False
    mock_kafka_consumer.return_value = consumer_instance

    tp_instance = MagicMock()
    mock_topic_partition.return_value = tp_instance

    consumer = create_kafka_consumer(
        settings=mock_settings,
        client_id="test-client",
        topic="test-topic",
        partition=0,
        offset=10,
        logger=logger,
    )

    mock_topic_partition.assert_called_once_with("test-topic", 0)
    consumer_instance.assign.assert_called_once_with([tp_instance])
    consumer_instance.seek.assert_called_once_with(tp_instance, 10)
    assert logger.info.call_count == 2
    assert consumer == consumer_instance



@patch("src.utils.kafka.KafkaProducer")
def test_create_kafka_producer_success(mock_kafka_producer):
    logger = MagicMock()
    settings = MagicMock()
    producer_instance = MagicMock()
    mock_kafka_producer.return_value = producer_instance

    producer = create_kafka_producer(settings=settings, logger=logger)

    mock_kafka_producer.assert_called_once()
    assert producer == producer_instance


@patch("src.utils.kafka.KafkaProducer", side_effect=NoBrokersAvailable())
def test_create_kafka_producer_no_broker(mock_kafka_producer):
    logger = MagicMock()
    settings = MagicMock()
    with pytest.raises(ConfigurationError):
        create_kafka_producer(settings=settings, logger=logger)


