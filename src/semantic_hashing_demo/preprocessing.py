"""
Preprocess the data and generate the hash buckets for the text reviews.
Store the buckets for different nbits into different files.
"""

import polars as pl
from config import data_file, embedding_size, model, seed
from lsh import LSH
from utils import ensure_file_exists

dir_name = "output"
file_name = "preprocessed_data.csv"


def main():
    # load data
    df = pl.read_csv(data_file)
    reviews = df.get_column("Text").to_list()

    # Generate embeddings for each review
    embeddings = LSH.get_embedding(reviews, model)

    reviews_updated = [
        review.replace("\n", " ").replace("<br />", " ") for review in reviews
    ]

    # Create DataFrame with updated reviews and embeddings
    df2 = pl.DataFrame(
        {
            "Text": reviews_updated,
            "Embedding": [str(embedding) for embedding in embeddings.tolist()],
        }
    )

    print("Writing Embeddings, Hashes, Buckets to CSV files...\n")
    for nbits in [8, 16, 32, 64, 128]:
        # Create LSH instance
        lsh = LSH(nbits=nbits, seed=seed, embedding_size=embedding_size)

        # Hash each embeddings into a hash code. Hence, a list of hash codes is returned
        hashes = lsh.hash_vector(embeddings)

        # Add LSH hashes corresponding to the embeddings to the df2 DataFrame
        df2.insert_column(len(df2.columns), pl.Series(f"Hash {nbits}-bit", hashes))

        # Hashes to buckets
        buckets = lsh.bucket_hashes(hashes)

        # Define the path for the directory and ensure the file
        bucket_file_name = f"buckets_{nbits}bit.csv"
        ensure_file_exists(dir_name, f"buckets_{nbits}bit.csv")

        # write to CSV
        lsh.write_buckets_to_csv(
            buckets, "Text Hash", "Text Indices", f"{dir_name}/{bucket_file_name}"
        )

        print(f"\tfor nbits = {nbits} ✅\n")

    # Define the path for the directory and the file
    ensure_file_exists(dir_name, file_name)

    """ Save embeddings + LSH to CSV, linked with source sample """
    df2.write_csv(f"{dir_name}/{file_name}", separator=",")


if __name__ == "__main__":
    main()
