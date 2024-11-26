import atexit

from pfolio.cli import pfolio_group


def exit_cli():
    """Application Exitpoint."""
    print("Cleanup actions here...")


def run_cli() -> None:
    """Application Entrypoint."""
    # atexit.register(exit_cli)
    pfolio_group(obj={})


if __name__ == '__main__':
    run_cli()