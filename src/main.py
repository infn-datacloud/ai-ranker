from src.inference.main import run as inference_run
from src.logger import create_logger
from src.parser import parser
from src.training.main import run as training_run


def main():
    args = parser.parse_args()
    logger = create_logger("ai-ranker", args.loglevel.upper())
    if args.training:
        training_run(logger)
    elif args.inference:
        inference_run(logger)
    else:
        logger.warning("Neither --training or --inference arguments have been defined")


if __name__ == "__main__":
    main()
