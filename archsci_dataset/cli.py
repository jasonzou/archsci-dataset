# archsci-dataset/cli.py

from typing import Annotated
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
    
@app.command("convert")
def convert(output_dir: Annotated[
        Path,
        typer.Option( 
            "--destionation", 
            "-d", 
            help="Output direcotry for processed data",
            dir_okay=True,
            writable=True,
            resolve_path=True,
        )
    ]="./output",
    endpoint: Annotated[
        str, 
        typer.Option( 
            "--endpoint", 
            "--endpoint", 
            "-e", 
            help="Ollama API endpoint", 
        )
    ]="http://localhost:11434", 
) -> None:
    print(endpoint)
    print(output_dir)
    print("jkfld;afsd")
    return

@app.callback()
def main(
    version: Annotated[
        bool, 
        typer.Option(
            "--version", 
            "-v", 
            help="Show the application's version and exit.", 
            callback=_version_callback, 
            is_eager=True,
    )]=False,
    output_dir: Annotated[
        Path,
        typer.Option( 
            "--destionation", 
            "-d", 
            help="Output direcotry for processed data",
            dir_okay=True,
            writable=True,
            resolve_path=True,
        )
    ]="./output",
    endpoint: Annotated[
        str, 
        typer.Option( 
            "--endpoint", 
            "-e", 
            help="Ollama API endpoint", 
        )
    ]="http://localhost:11434", 
) -> None:
    print(output_dir)
    print(endpoint)
    return