from typing import Dict, List

import numpy as np
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
    plane_norms = np.random.rand(nbits, len(v)) - 0.5

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
    bucket = {}

    for i, hash_str in enumerate(v):
        if hash_str in bucket:
            bucket[hash_str].append(i)
        else:
            bucket[hash_str] = [i]

    return bucket


def main():
    """
    Categorize the search queries into buckets based on their hash values using LSH random projection approach.
    """

    # Search queries
    queries = [
        "How to get started with machine learning",
        "How to get started with machine learning",
        "How to get started with machine learning",
        "How to get started with machine learning",
        # "How to get started with deep learning",
        # "How to get started with computer vision",
        # "How to get started with natural language processing",
    ]
    print("\nqueries")
    print(queries)

    # embeddings vector for each query
    embeddings = [get_embedding(query) for query in queries]

    nbits = 2  # no. of hyperplanes
    hashed_vectors = [hash_vector(embedding, nbits) for embedding in embeddings]
    print("\nhashed vectors")
    print(hashed_vectors)

    # TODO: Convert the embeddings to numpy array
    print("\nbucket")
    bucket = bucket_hashes(hashed_vectors)
    print(bucket)


# end def

if __name__ == "__main__":
    main()
