from pathlib import Path

import click
import pandas as pd
import pandas_profiling


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-o",
    "--output_dir",
    default="data/profiles",
    type=click.Path(file_okay=False, writable=True, path_type=Path),
    show_default=True,
)
def profile_data(dataset_path: Path, output_dir: Path):
    if dataset_path.suffix != ".csv":
        raise TypeError(
            "Data profiling does not support anything other than .csv data extension"
        )
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    profile_name = f"profile_{dataset_path.stem}_{dataset_path.suffix[1:]}"
    pandas_profiling.ProfileReport(pd.read_csv(dataset_path)).to_file(
        output_dir / f"{profile_name}.html"
    )
