from inference.main import load_data_from_file
from logger import create_logger
from settings import load_inference_settings
from utils.kafka import create_kafka_producer

logger = create_logger("Populate inference topic", level="INFO")
settings = load_inference_settings(logger=logger)
producer = create_kafka_producer(
    kafka_server_url=settings.KAFKA_HOSTNAME, logger=logger
)
messages = load_data_from_file(filename=settings.LOCAL_IN_MESSAGES, logger=logger)
for i, message in enumerate(messages):
    producer.send(settings.KAFKA_INFERENCE_TOPIC, message)
    producer.flush()
    logger.info("Message sent (%d/%d)", i + 1, len(messages))
    input("Waiting to send next message...")
