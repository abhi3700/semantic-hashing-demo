from typing import Dict, List

import numpy as np
import polars as pl
from config import data_file, model, n, nbits, seed
from numpy.random import RandomState
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


def hash_vector(v: List[np.float64], nbits: np.uint16, plane_norms) -> str:
    """LSH random projection hash function with seeded hyperplane generation.

    Args:
        v (List[np.float64]): embedding vector.
        nbits (np.uint16): no. of hyperplanes.
        seed (str, optional): Seed for the random number generator. Defaults to 'subspace'.

    Returns:
        str: A binary string of length nbits.
    """

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


def contains_zero(arr) -> bool:
    """Check if the given array contains zero."""
    return 0 in arr


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
    embeddings = [get_embedding(info, model) for info in infos]

    # Generate hyperplanes using the seeded random number generator
    # Create a RandomState instance with the seed
    rng = RandomState(seed)
    plane_norms = rng.rand(int(nbits), len(embeddings[0])) - 0.5

    # hash the embeddings vector
    hashed_vectors = [
        hash_vector(embedding, nbits, plane_norms) for embedding in embeddings
    ]
    print("\nhashed vectors:")
    print(hashed_vectors)

    # Bucket the hashed vectors
    bucket = bucket_hashes(hashed_vectors)
    print(f"\nbuckets has a length of {len(bucket.keys())}")
    print(bucket)

    # =============== B. Bucketing of a given text into available buckets===============
    # search query
    # ===== Type-1 =====
    query = infos[0]  # try with the 1st one to verify the correctness
    # query = infos[1]  # try with the 1st one to verify the correctness

    # ===== Type-2 =====
    # query = "I have bought many of the Vitality canned dog food products and have found them all to be of good quality. The product looks more like a stew than a processed meat and it smells good. My Labrador is finicky and she likes this product better than  most."  # changed the 1st review a bit
    # query = 'Product reached marked as Jumbo Salted Peanuts...the peanuts were actually small sized unsalted. Not sure if this was a mistake or if the vendor wanted to indicate the product as "Jumbo".'  # changed the 2nd review a bit
    # query = 'Great taffy at a better price.  There was a broad assortment of yummy taffy.  Delivery was super fast.  If your a taffy lover, this is a good chance.'  # changed the 5th review a bit

    # ===== Type-3 =====
    # query = "I've purchased numerous cans of the Vitality dog food line and have consistently found them to be of high quality. They resemble stew more than they do processed meat, and they have a more pleasant aroma. My picky Labrador prefers this brand over many others." # ai-generated the 1st review a bit
    # query = 'Excellent value for delicious taffy. The selection offered a broad variety of delectable flavors. The shipping was impressively fast. For enthusiasts of taffy, this offer is a must-grab.'    # ai-generated the 5th review a bit
    # query = "This dog food is highly nutritious and beneficial for digestive health. It's also suitable for young puppies. My dog consistently consumes the recommended portion at each meal."  # ai-generated the index-9 review a bit

    # ===== Type-4 =====
    # query = "I have lived out of the US for over 7 yrs now, and I so miss my Twizzlers!!  When I go back to visit or someone visits me, I always stock up.  All I can say is YUM!<br />Sell these in Mexico and you will have a faithful buyer, more often than I'm able to buy them right now."
    # query = "Good oatmeal.  I like the apple cinnamon the best.  Though I wouldn't follow the directions on the package since it always comes out too soupy for my taste.  That could just be me since I like my oatmeal really thick to add some milk on top of."
    # query = "I roast at home with a stove-top popcorn popper (but I do it outside, of course). These beans (Coffee Bean Direct Green Mexican Altura) seem to be well-suited for this method. The first and second cracks are distinct, and I've roasted the beans from medium to slightly dark with great results every time. The aroma is strong and persistent. The taste is smooth, velvety, yet lively."
    hash_query = hash_vector(get_embedding(query), nbits, plane_norms)
    # hash_query = hash_vector(get_embedding(query, model), nbits, plane_norms)
    print(f"\nFor a given text: \"{query}\", it's computed hash is '{hash_query}'.")

    # calculate the hamming distance between the query and each bucket
    print("\nhamming distances b/w the query from each bucket key:")
    hamming_distances = []
    for hash_str in bucket.keys():
        hamming_distances.append(hamming_distance(hash_query, hash_str))
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
    min_index = np.argmin(hamming_distances)
    print(
        f"\nHence, the given text belongs to the index-{min_index} of bucket list, \ni.e. the bucket with key: '{list(bucket.keys())[min_index]}', value: [{list(bucket.values())[min_index]}]."
    )


# end def

if __name__ == "__main__":
    main()
