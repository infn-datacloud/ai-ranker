from parser import parser

from logger import create_logger
from training.main import run as training_run

# from inference.main import run as inference_run


if __name__ == "__main__":
    args = parser.parse_args()
    logger = create_logger("ai-ranker", args.loglevel.upper())
    if args.training:
        training_run(logger)
    elif args.inference:
        pass
        # inference_run(logger)
    logger.warning("Neither --training or --inference arguments have been defined")