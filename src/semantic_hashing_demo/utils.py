import pathlib


def check_file_exists(dir: pathlib.Path, file_path: pathlib.Path):
    # Create the directory if it doesn't exist
    if not dir.exists():
        dir.mkdir(parents=True, exist_ok=True)

    # Create the file if it doesn't exist
    if not file_path.is_file():
        with file_path.open("w") as _:
            pass
