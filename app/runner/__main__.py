import argparse
from .cli import run_from_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path al YAML de configuración")
    args = ap.parse_args()
    run_from_yaml(args.config)


if __name__ == "__main__":
    main()
