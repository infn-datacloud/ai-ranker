from parser import parser
from training.main import run as training_run
# from inference.main import run as inference_run


if __name__ == "__main__":
    args = parser.parse_args()
    if args.training:
        training_run()
    elif args.inference:
        pass
        # inference_run()
    print("Neither --training or --inference arguments have been defined")