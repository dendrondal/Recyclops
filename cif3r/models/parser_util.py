import os
import argparse


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-nep", "--epochs", type=int, help="number of epochs to train for", default=100
    )

    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        help="learning rate for the model, default=0.001",
        default=0.001,
    )

    parser.add_argument(
        "-lrS",
        "--lr_scheduler_step",
        type=int,
        help="StepLR learning rate scheduler step, default=20",
        default=20,
    )

    parser.add_argument(
        "-lrG",
        "--lr_scheduler_gamma",
        type=float,
        help="StepLR learning rate scheduler gamma, default=0.5",
        default=0.5,
    )

    parser.add_argument(
        "-its",
        "--iterations",
        type=int,
        help="number of episodes per epoch, default=100",
        default=100,
    )

    parser.add_argument(
        "-cTr",
        "--classes_per_it_tr",
        type=int,
        help="number of random classes per episode for training, default=15",
        default=13,
    )

    parser.add_argument(
        "-nsTr",
        "--num_support_tr",
        type=int,
        help="number of samples per class to use as support for training, default=5",
        default=4,
    )

    parser.add_argument(
        "-nqTr",
        "--num_query_tr",
        type=int,
        help="number of samples per class to use as query for training, default=5",
        default=3,
    )

    parser.add_argument(
        "-cVa",
        "--classes_per_it_val",
        type=int,
        help="number of random classes per episode for validation, default=5",
        default=10,
    )

    parser.add_argument(
        "-nsVa",
        "--num_support_val",
        type=int,
        help="number of samples per class to use as support for validation, default=5",
        default=3,
    )

    parser.add_argument(
        "-nqVa",
        "--num_query_val",
        type=int,
        help="number of samples per class to use as query for validation, default=15",
        default=4,
    )

    parser.add_argument(
        "-seed",
        "--manual_seed",
        type=int,
        help="input for the manual seeds initializations",
        default=42,
    )

    parser.add_argument(
        "-uni",
        "--university",
        type=str,
        help="University guidelines to use",
        default="UTK",
    )
    return parser
