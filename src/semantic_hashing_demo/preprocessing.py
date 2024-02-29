""" 
Preprocess the data and generate the hash buckets for the text reviews.
Store the buckets for different nbits into different files.
"""

import pathlib

import polars as pl
from config import data_file, embedding_size, model, seed
from lsh import LSH
from utils import check_file_exists

output_dir = pathlib.Path("output")


def main():
    # load data
    df = pl.read_csv(data_file)
    reviews = df.get_column("Text").to_numpy().flatten()

    print("Writing buckets to CSV files...\n")
    for nbits in [8, 16, 32, 64, 128]:
        # Create LSH instance
        lsh = LSH(nbits=nbits, seed=seed, embedding_size=embedding_size)

        # Generate embeddings for each review
        embeddings = lsh.get_embedding(reviews, model)

        # Hash each embeddings into a hash code. Hence, a list of hash codes is returned
        hashes = lsh.hash_vector(embeddings)

        # Hashes to buckets
        buckets = lsh.bucket_hashes(hashes)

        # Define the path for the directory and ensure the file
        bucket_file_name = f"buckets_{nbits}bit.csv"
        check_file_exists(output_dir, output_dir / bucket_file_name)

        # write to CSV
        lsh.write_buckets_to_csv(
            buckets, "Text Hash", "Text Indices", output_dir / bucket_file_name
        )

        print(f"\tfor nbits = {nbits} ✅\n")


if __name__ == "__main__":
    main()
