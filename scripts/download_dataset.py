import argparse

from src.data.download import download_cc100, download_oscar, download_wikipedia


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wikipedia", action="store_true", default=False)
    parser.add_argument("--cc100", action="store_true", default=False)
    parser.add_argument("--oscar", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.wikipedia:
        download_wikipedia(args.seed)
    if args.cc100:
        download_cc100(args.seed)
    if args.oscar:
        download_oscar(args.seed)


if __name__ == "__main__":
    main()
