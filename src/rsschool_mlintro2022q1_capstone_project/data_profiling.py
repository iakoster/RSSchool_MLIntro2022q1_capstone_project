from pathlib import Path

import pandas as pd


def profile_data():
    dataset_path = Path() / r'data\train.csv'
    output_dir = Path() / r'data\profiles'

    if dataset_path.suffix != '.csv':
        raise TypeError(
            'Data profiling does not support '
            'anything other than .csv data type')
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    profile_name = f'profile_{dataset_path.stem}_' \
                   f'{dataset_path.suffix[1:]}'
    pd.read_csv(dataset_path)\
        .profile_report(title=profile_name)\
        .to_file(output_dir / f'{profile_name}.html')