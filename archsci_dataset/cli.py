# archsci-dataset/cli.py

from typing import Optional, Annotated
import typer
from pathlib import Path

from archsci_dataset import __app_name__, __version__

app = typer.Typer(
    help="Archival Science dataset tool",
    no_args_is_help = True,
    )

@app.command("version")
def _version_callback(value: bool = True) -> None:
    if value:
        typer.echo(f"{__app_name__} v{__version__}")
        raise typer.Exit()

@app.callback()
def main(
    root: Annotated[
        Path,
        typer.Option(
            help="The project root directory.",
            dir_okay=True,
            writable=True,
            resolve_path=True,
        ),
    ]="./output",
    force: Annotated[
        bool,
        typer.Option(help="Force initialization even if the project already exists."),
    ] = False,
    version: Annotated[bool, 
        typer.Option("--version", "-v", help="Show the application's version and exit.", callback=_version_callback, is_eager=True,
    )]=False,
    # output_dir: Optional[str] = typer.Option(
    #     "output",
    #     "--destionation",
    #     "-d",
    #     help="Output direcotry for processed data",
    #     is_eager=True,
    # ),
    # endpoint: Optional[str] = typer.Option(
    #     "http://localhost:11434",
    #     "--endpoint",
    #     "-e",
    #     help="Ollama API endpoint",
    #     is_eager=True,
    # )
) -> None:
    # print(output_dir)
    # print(endpoint)
    return