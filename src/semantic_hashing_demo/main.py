from typing import Dict, List

import numpy as np
import polars as pl
from input import data_file, n, nbits
from openai import OpenAI

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
    """LSH random projection hash function.
    It is a simple hash function that takes a vector and returns a binary string of length nbits.

    Args:
        v (List[np.float64]): embedding vector.
        nbits (np.uint16): no. of hyperplanes.

    Returns:
        str: A binary string of length nbits.
    """
    # create a set of 4 hyperplanes (normal ⟂), with 2 dimensions
    plane_norms = np.random.rand(int(nbits), len(v)) - 0.5

    v_np = np.asarray(v)

    # calculate the dot product for each of these
    v_dot = np.dot(v_np, plane_norms.T)

    # we know that a positive dot product == +ve side of hyperplane
    # and negative dot product == -ve side of hyperplane
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
    bucket: Dict = {}

    for i, hash_str in enumerate(v):
        if hash_str in bucket:
            bucket[hash_str].append(i)
        else:
            bucket[hash_str] = [i]

    return bucket


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
    query = infos[0]  # try with the 1st one to verify the correctness
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
        f"\nHence, the given text belongs to the {min_index}th index of the bucket with key: {list(bucket.keys())[min_index]}."
    )


# end def

if __name__ == "__main__":
    main()
