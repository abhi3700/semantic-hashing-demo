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
    reviews = reviews[0:10]

    for nbits in [8, 16, 32, 64, 128]:
        print(f"\n\n=====For nbits = {nbits}======\n")

        # Create LSH instance
        lsh = LSH(nbits=nbits, seed=seed, embedding_size=embedding_size)

        # Generate embeddings for each review
        embeddings = [LSH.get_embedding(review, model) for review in reviews]

        # Hash the embeddings
        hashed = [lsh.hash_vector(embedding) for embedding in embeddings]

        # Bucket the hashes
        buckets_df = lsh.hashes_to_df(hashed, "Text Hash", "Text Indices")
        print(buckets_df)

        # Define the path for the directory and ensure the file
        bucket_file_name = f"buckets_{nbits}bit.csv"
        check_file_exists(output_dir, output_dir / bucket_file_name)

        # FIXME: write to CSV
        # buckets_df.write_csv(output_dir / bucket_file_name, separator=",")


if __name__ == "__main__":
    main()
