from logger import create_logger
from settings import load_inference_settings
from utils.kafka import create_kafka_consumer

logger = create_logger("Read ranked providers", level="INFO")
settings = load_inference_settings(logger=logger)
consumer = create_kafka_consumer(
    kafka_server_url=settings.KAFKA_HOSTNAME,
    topic=settings.KAFKA_RANKED_PROVIDERS_TOPIC,
    logger=logger,
)
for message in consumer:
    logger.info("Received message: %s", message)
