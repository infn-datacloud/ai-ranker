from logger import create_logger
from settings import load_inference_settings
from utils.kafka import create_kafka_consumer

logger = create_logger("Read training", level="INFO")
settings = load_inference_settings(logger=logger)
consumer = create_kafka_consumer(
    kafka_server_url=settings.KAFKA_HOSTNAME,
    topic=settings.KAFKA_TRAINING_TOPIC,
    logger=logger,
)
for message in consumer:
    logger.info("Received message: %s", message)
