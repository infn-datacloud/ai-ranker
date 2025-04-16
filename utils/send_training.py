from logger import create_logger
from settings import load_training_settings
from utils import MSG_VERSION, load_local_dataset
from utils.kafka import create_kafka_producer

logger = create_logger("Populate training", level="INFO")
settings = load_training_settings(logger=logger)
producer = create_kafka_producer(
    kafka_server_url=settings.KAFKA_HOSTNAME, logger=logger
)
df = load_local_dataset(
    filename=settings.LOCAL_DATASET,
    dataset_version=settings.LOCAL_DATASET_VERSION,
    logger=logger,
)
for index, row in df.iterrows():
    producer.send(
        settings.KAFKA_TRAINING_TOPIC,
        {MSG_VERSION: settings.LOCAL_DATASET_VERSION, **row.to_dict()},
    )
    logger.info("Message sent (%d/%d)", index + 1, len(df))
producer.flush()
