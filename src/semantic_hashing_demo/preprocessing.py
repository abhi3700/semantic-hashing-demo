""" 
Demo-2 is about storing embeddings & semantic hashes in a CSV file.
"""

import pathlib

import polars as pl
from config import data_file, model
from main import get_embedding, hash_vector, seed, update_text
from numpy.random import RandomState

OUTPUT_DIR_NAME = "output"
EMBEDDINGS_FILE_NAME = "preprocessed_data.csv"


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
    """
    For each sample in dataset (use the 1000 reviews)
        1. Create embedding of sample
        2. Save embeddings to disk, linked with source sample
    """
    """a. create embedding of sample"""
    # pull data into polars dataframe
    df = pl.read_csv(data_file)
    reviews = df.select(pl.col("Text")).to_numpy().flatten()

    # remove <br />, \n
    reviews_updated = []
    # for usage in LSH later.
    embeddings = []
    # for storing in CSV file
    embeddings_str = []

    for review in reviews:
        reviews_updated.append(update_text(review))
        embedding = get_embedding(review, model)
        embeddings.append(embedding)
        embeddings_str.append(str(embedding))

    df2 = pl.DataFrame({"Text": reviews_updated, "Embedding": embeddings_str})

    # Define the path for the directory and the file
    output_dir = pathlib.Path(OUTPUT_DIR_NAME)
    file_path = output_dir / EMBEDDINGS_FILE_NAME
    check_file_exists(output_dir, file_path)

    """ b. Save embeddings to CSV, linked with source sample """
    # df2.write_csv(file_path, separator=",")

    # =============== Process-2 ===============
    """ 
    for each nbits = [8, 16, 32, 64, 128]
        1. Apply LSH to the embedding
        2. Add hash to a hash table, along with index of embedding
        3. Save hashes to disk (categorize by parameter)    
    """
    # For LSH, generate hyperplanes using the seeded random number generator
    # Create a RandomState instance with the seed
    rng = RandomState(seed)
    e_len = len(embeddings[0])

    nbits_list = [8, 16, 32, 64, 128]
    for i, nbits in enumerate(nbits_list):
        # get hyperplanes for nbits
        plane_norms = rng.rand(nbits, e_len) - 0.5

        # Apply LSH to the embedding
        hashed_vectors: list[str] = [
            hash_vector(embedding, plane_norms) for embedding in embeddings
        ]

        df2.insert_column(
            i + 2, pl.Series("Hash {nbits}-bit".format(nbits=nbits), hashed_vectors)
        )

    """ Save embeddings + LSH to CSV, linked with source sample """
    df2.write_csv(file_path, separator=",")


if __name__ == "__main__":
    main()
