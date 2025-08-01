from src.logger import create_logger
from src.settings import load_training_settings
from src.utils import MSG_VERSION, load_local_dataset
from src.utils.kafka import create_kafka_producer

logger = create_logger("Populate training", level="INFO")
settings = load_training_settings(logger=logger)
producer = create_kafka_producer(settings=settings, logger=logger)
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
