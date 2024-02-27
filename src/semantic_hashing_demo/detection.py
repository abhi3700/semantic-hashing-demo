""" 
The end goal is to detect AI-generated text.
"""
import numpy as np
import polars as pl
from config import model, preprocessed_data_file, seed
from main import contains_zero, get_embedding, hamming_distance, hash_vector
from numpy.random import RandomState

query = "I have bought many of the Vitality canned dog food products and have found them all to be of good quality. The product looks more like a stew than a processed meat and it smells good. My Labrador is finicky and she likes this product better than  most."


def main():
    nbits_list = [8, 16, 32, 64, 128]

    # Specify data types for multiple columns to be read as strings
    dtype_spec = {f"Hash {nbits}-bit": str for nbits in nbits_list}
    # pull data into polars dataframe
    df = pl.read_csv(preprocessed_data_file, dtypes=dtype_spec)
    # reviews = df.select(pl.col("Text")).to_numpy().flatten()
    embeddings = df.select(pl.col("Embedding")).to_numpy().flatten()
    e_len = len(eval(embeddings[0]))

    # Generate hyperplanes using the seeded random number generator
    # Create a RandomState instance with the seed
    rng = RandomState(seed)

    # buckets for all nbits
    dtype_spec = {"Hash": str, "Review Indices": str}
    for nbits in nbits_list:
        print(f"\n\n=====For nbits = {nbits}======\n")

        # plane norms for this nbits
        plane_norms = rng.rand(int(nbits), e_len) - 0.5

        # get the hash of query
        query_hash = hash_vector(get_embedding(query, model), plane_norms)
        print(
            f"\nFor a given text: \n\"{query}\", \nit's computed hash is '{query_hash}'."
        )

        df2 = pl.read_csv(f"output/buckets_{nbits}bit.csv", dtypes=dtype_spec)

        df2 = df2.filter(pl.col("Hash") == query_hash)
        bucket_key_hashes = df2.select(pl.col("Hash")).to_numpy().flatten()
        bucket_review_indices = (
            df2.select(pl.col("Review Indices")).to_numpy().flatten()
        )
        # print(bucket_review_indices)

        # calculate the hamming distance between the query and each bucket
        print("\nhamming distances b/w the query from each bucket key:")
        hamming_distances = []
        for hash_str in bucket_key_hashes:
            hamming_distances.append(hamming_distance(query_hash, hash_str))
        print(hamming_distances)
        if contains_zero(hamming_distances):
            print(
                "🙂 The given text falls into the bucket with its key having exact same hash"
            )
        else:
            print(
                "☹️ As no exact hash found, deliberately the closest bucket with min. hamming distance is selected here from left --> right."
            )
        # Get the index of the lowest one
        if len(hamming_distances) > 0:
            min_index = np.argmin(hamming_distances)
            print(
                f"\nHence, the given text belongs to the index-{min_index} of bucket list, \ni.e. the bucket with key: '{bucket_key_hashes[min_index]}', value: [{bucket_review_indices[min_index]}]."
            )


if __name__ == "__main__":
    main()
