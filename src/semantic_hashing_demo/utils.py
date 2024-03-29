from pathlib import Path
from typing import List


def ensure_file_exists(dir_name: str, file_name: str) -> None:
    """
    Create the directory and the file if it doesn't exist.
    The function is efficient as it avoids redundant directory and file creation.

    Args:
        dir (Path): The directory path.
        file_name (str): The file name to create.
    """
    dir = Path(dir_name)
    file_path = dir.joinpath(file_name)
    if not dir.exists():
        dir.mkdir(parents=True, exist_ok=True)
    elif not file_path.is_file():
        file_path.touch()


def check_files_exist(directory: str, file_names: List[str]) -> bool:
    """
    Check if files exist in the given directory.

    Args:
        directory (str): The directory path.
        file_names (List[str]): List of file names to check.

    Returns:
        bool: If all of the files exist in the directory.
    """
    return all(Path(directory).joinpath(name).is_file() for name in file_names)
