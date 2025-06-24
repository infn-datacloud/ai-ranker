from kafka.admin import KafkaAdminClient

from src.logger import create_logger
from src.settings import load_inference_settings

logger = create_logger("Delete ranked provider", level="INFO")
settings = load_inference_settings(logger=logger)
admin_client = KafkaAdminClient(bootstrap_servers=settings.KAFKA_HOSTNAME)
admin_client.delete_topics(topics=[settings.KAFKA_RANKED_PROVIDERS_TOPIC])
