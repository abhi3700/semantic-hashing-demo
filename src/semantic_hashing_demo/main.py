from typing import Dict, List

import numpy as np
import polars as pl
from numpy.random import RandomState
from openai import OpenAI

from semantic_hashing_demo.config import data_file, n, nbits, seed

client = OpenAI()


def get_embedding(text: str, model="text-embedding-3-small"):
    """Get the embedding vector of a given text with default OpenAI embedding small model.
    Small embedding model: 1536 len of float values.
    Large embedding model: 3072 len of float values.

    Args:
        text (str): search query/reviews/comments.
        model (str, optional): supported embedding model. Defaults to "text-embedding-3-small".

    Returns:
        CreateEmbeddingResponse: list of float values.
    """
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def hash_vector(v: List[np.float64], nbits: np.uint16) -> str:
    """LSH random projection hash function with seeded hyperplane generation.

    Args:
        v (List[np.float64]): embedding vector.
        nbits (np.uint16): no. of hyperplanes.
        seed (str, optional): Seed for the random number generator. Defaults to 'subspace'.

    Returns:
        str: A binary string of length nbits.
    """
    # Create a RandomState instance with the seed
    # rng = RandomState(seed_int)
    rng = RandomState(seed)

    # Generate hyperplanes using the seeded random number generator
    plane_norms = rng.rand(int(nbits), len(v)) - 0.5

    v_np = np.asarray(v)
    v_dot = np.dot(v_np, plane_norms.T)
    v_dot = v_dot > 0
    v_dot = v_dot.astype(int)
    v_dot = "".join(str(i) for i in v_dot)

    return v_dot


def bucket_hashes(v: List[str]) -> Dict[str, List[np.uint8]]:
    """Distribute hashes into corresponding buckets

    Args:
        v (List[str]): list of hash strings.

    Returns:
        Dict[str, List[np.uint8]]: Buckets as hashmap where key is hash string and value is list of indices.
                                    Each index corresponds to the original query
    """
    buckets: Dict = {}

    for i, hash_str in enumerate(v):
        # create bucket if it doesn't exist
        if hash_str not in buckets.keys():
            buckets[hash_str] = []

        # add vector position to bucket
        buckets[hash_str].append(i)

    return buckets


def hamming_distance(str1: str, str2: str) -> int:
    """Calculate the Hamming distance between two strings.

    Args:
        str1 (str): The first string.
        str2 (str): The second string.

    Returns:
        int: The Hamming distance between the two strings.

    Raises:
        ValueError: If the lengths of the strings are not equal.
    """
    if len(str1) != len(str2):
        raise ValueError("Strings must be of equal length")

    distance = 0
    for char1, char2 in zip(str1, str2):
        if char1 != char2:
            distance += 1

    return distance


def main():
    """
    Print the index of the bucket that the given text belongs to.
    """
    # =============== A. Bucketing of Texts ===============
    print(f"Parsing {n} samples with {nbits} hyperplanes for bucketing...")

    # knowledge base
    df = pl.read_csv(data_file)
    reviews = df.select("Text").to_numpy().flatten()
    # get text samples
    infos = reviews[:n]
    print("\nInformation or Knowledge base (1st 5 samples):")
    print(infos[0:5])

    # embeddings vector for each info
    embeddings = [get_embedding(info) for info in infos]

    # hash the embeddings vector
    hashed_vectors = [hash_vector(embedding, nbits) for embedding in embeddings]
    print("\nhashed vectors:")
    print(hashed_vectors)

    # Bucket the hashed vectors
    bucket = bucket_hashes(hashed_vectors)
    print(f"\nbuckets has a length of {len(bucket.keys())}")
    print(bucket)

    # =============== B. Bucketing of a given text into available buckets===============
    # search query
    # query = infos[0]  # try with the 1st one to verify the correctness
    # query = infos[1]  # try with the 1st one to verify the correctness
    # query = "I have bought many of the Vitality canned dog food products and have found them all to be of good quality. The product looks more like a stew than a processed meat and it smells good. My Labrador is finicky and she likes this product better than  most."  # changed the 1st review a bit
    query = 'Product reached marked as Jumbo Salted Peanuts...the peanuts were actually small sized unsalted. Not sure if this was a mistake or if the vendor wanted to indicate the product as "Jumbo".'  # changed the 2nd review a bit
    hash_query = hash_vector(get_embedding(query), nbits)
    print(f"\nFor a given text: \"{query}\", it's computed hash is '{hash_query}'.")

    # calculate the hamming distance between the query and each bucket
    print("\nhamming distances b/w the query from each bucket key:")
    hamming_distances = []
    for hash_str in bucket.keys():
        hamming_distances.append(hamming_distance(hash_query, hash_str))
    print(hamming_distances)

    # Get the index of the lowest one
    min_index = np.argmin(hamming_distances)
    print(
        f"\nHence, the given text belongs to the {min_index}th index of the bucket with key: '{list(bucket.keys())[min_index]}', value: [{list(bucket.values())[min_index][0]}]."
    )


# end def

if __name__ == "__main__":
    main()
