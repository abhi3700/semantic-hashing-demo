""" 
Demo-2 is about storing embeddings & semantic hashes in a CSV file.
"""

import pathlib

import polars as pl
from config import data_file, model
from main import get_embedding, update_text

OUTPUT_DIR_NAME = "output"
EMBEDDINGS_FILE_NAME = "embeddings.csv"


def check_file_exists(dir: pathlib.Path, file_path: pathlib.Path):
    # Create the directory if it doesn't exist
    if not dir.exists():
        dir.mkdir(parents=True, exist_ok=True)

    # Create the file if it doesn't exist
    if not file_path.is_file():
        with file_path.open("w") as _:
            pass


def main():
    # =============== Process-1 ===============
    """a. create embedding of sample"""
    # pull data into polars dataframe
    df = pl.read_csv(data_file)
    reviews = df.select("Text").to_numpy().flatten()

    reviews_updated = []
    embeddings = []

    for review in reviews:
        reviews_updated.append(update_text(review))
        embeddings.append(str(get_embedding(review, model)))

    df2 = pl.DataFrame({"Text": reviews_updated, "Embedding": embeddings})

    # Define the path for the directory and the file
    output_dir = pathlib.Path(OUTPUT_DIR_NAME)
    file_path = output_dir / EMBEDDINGS_FILE_NAME
    check_file_exists(output_dir, file_path)

    """ b. Save embeddings to CSV, linked with source sample """
    df2.write_csv(file_path, separator=",")


if __name__ == "__main__":
    main()
