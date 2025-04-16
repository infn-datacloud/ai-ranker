from kafka.admin import KafkaAdminClient

from logger import create_logger
from settings import load_inference_settings

logger = create_logger("Delete inference", level="INFO")
settings = load_inference_settings(logger=logger)
admin_client = KafkaAdminClient(bootstrap_servers=settings.KAFKA_HOSTNAME)
admin_client.delete_topics(topics=[settings.KAFKA_INFERENCE_TOPIC])
