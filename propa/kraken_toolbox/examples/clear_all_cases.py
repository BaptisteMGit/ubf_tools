#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   clear_all_cases.py
@Time    :   2026/09/11 14:09:57
@Author  :   Menetrier Baptiste 
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
'''

# ======================================================================================================================
# Import
# ======================================================================================================================
from pathlib import Path
import shutil


def clean_examples_directory(
    examples_dir: str | Path,
    dry_run: bool = False,
) -> None:
    """
    Clean the examples directory.

    Recursively:
        - Keep .py and .csv files.
        - Delete all other files.
        - Delete every directory named 'parallel_working_dir'
          and all its contents.

    Parameters
    ----------
    examples_dir : str | Path
        Path to the examples directory.

    dry_run : bool, optional
        If True, only display what would be deleted.
        If False, actually delete the files and directories.
        Default is False.
    """
    examples_dir = Path(examples_dir)

    if not examples_dir.exists():
        raise FileNotFoundError(
            f"Examples directory does not exist: {examples_dir}"
        )

    if not examples_dir.is_dir():
        raise NotADirectoryError(
            f"Path is not a directory: {examples_dir}"
        )

    allowed_extensions = [".py", ".csv"]

    # First remove all parallel_working_dir directories.
    for directory in examples_dir.rglob("parallel_working_dir"):
        if not directory.is_dir():
            continue

        action = "Would delete" if dry_run else "Deleted"
        print(f"{action}: {directory}")

        if not dry_run:
            shutil.rmtree(directory)

    # Then remove all files except .py and .csv.
    for file_path in examples_dir.rglob("*"):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in allowed_extensions:
            action = "Would delete" if dry_run else "Deleted"
            print(f"{action}: {file_path}")

            if not dry_run:
                file_path.unlink()


if __name__ == '__main__':
    clean_examples_directory("examples", dry_run=False)
