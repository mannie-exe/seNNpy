from os.path import abspath
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="seNNpy is the Discord bot of your hot, wet dreams"
    )
    parser.add_argument(
        "--train",
        type=int,
        action="store",
        help="if >1, training for that many epochs before continuing",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    if args.train:
        import training

        training.run(args.train)
    import generate
