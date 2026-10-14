"""E2T paper Full Model main entry point (Full Model exclusively)."""
from train import run
import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="brn_all", choices=("cev_all", "inu_all", "emd_all", "brn_all", "mro_all"))
    p.add_argument("--seeds", type=int, nargs="+", default=[3407,3408,3409,3410,3411])
    args = p.parse_args()
    run(dataset=args.dataset, seeds=tuple(args.seeds))


if __name__ == "__main__": main()
