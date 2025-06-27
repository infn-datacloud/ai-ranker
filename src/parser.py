import argparse
import logging

log_values = [i.lower() for i in logging._nameToLevel.keys()]

parser = argparse.ArgumentParser()

parser.add_argument(
    "-l",
    "--loglevel",
    default="info",
    choices=log_values,
    help=f"Provide logging level. Valid values: {log_values}. "
    "Example --loglevel debug, default=info",
)

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument(
    "-t", "--training", action="store_true", help="Start the training script."
)
group.add_argument(
    "-i", "--inference", action="store_true", help="Start the inference service."
)
