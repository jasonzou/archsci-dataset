# rptodo/__main__.py

from archsci_dataset import cli, __app_name__
from loguru import logger
import sys
config = {
    "handlers": [
        {
            "sink": sys.stdout,
            "format": "<blue>{time:YYYY-MM-DD HH:mm:ss}</blue> [{level}] {message}",
        },
        {
            "sink": "archsci_dataset.log",
            "format": "<blue>{time:YYYY-MM-DD HH:mm:ss}</blue> [{level}]\t{message}",
        }
    ],
}

logger.configure(**config)

def main():
    cli.app(prog_name=__app_name__)

if __name__ == "__main__":
    main()