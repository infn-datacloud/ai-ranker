from kafka.admin import KafkaAdminClient

from logger import create_logger
from settings import load_training_settings

logger = create_logger("Delete training", level="INFO")
settings = load_training_settings(logger=logger)
admin_client = KafkaAdminClient(bootstrap_servers=settings.KAFKA_HOSTNAME)
admin_client.delete_topics(topics=[settings.KAFKA_TRAINING_TOPIC])
