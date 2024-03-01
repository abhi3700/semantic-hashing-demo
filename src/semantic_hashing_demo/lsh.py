import csv
from typing import Dict, List

import numpy as np
import polars as pl

# from config import data_file, generated_data_file, model, nbits, seed
from openai import OpenAI


class LSH:
    def __init__(self, nbits: int, embedding_size: int, seed: int):
        self.nbits = nbits
        self.seed = seed
        self.plane_norms = self._generate_plane_norms(embedding_size)

    def _generate_plane_norms(self, embedding_size: int) -> np.ndarray:
        rng = np.random.RandomState(self.seed)
        return rng.rand(self.nbits, embedding_size) - 0.5

    @staticmethod
    def get_embedding(texts: List[str], model: str) -> np.ndarray:
        client = OpenAI()
        processed_texts = [
            text.replace("\n", " ").replace("<br />", " ") for text in texts
        ]
        embeddings = client.embeddings.create(input=processed_texts, model=model).data
        return np.array([embedding.embedding for embedding in embeddings])

    def hash_vector(self, v: np.ndarray) -> List[str]:
        v_dots = np.dot(v, self.plane_norms.T) > 0
        return ["".join(str(int(i)) for i in v_dot) for v_dot in v_dots]

    @staticmethod
    def bucket_hashes(v: List[str]) -> Dict[str, List[int]]:
        buckets = {}
        for idx, hash_str in enumerate(v):
            buckets.setdefault(hash_str, []).append(idx)
        return buckets

    @staticmethod
    def hashes_to_df(v: List[str], col1: str, col2: str) -> pl.DataFrame:
        buckets = {}
        for i, hash_str in enumerate(v):
            buckets.setdefault(hash_str, []).append(i)

        buckets_df = pl.from_dict(
            {col1: list(buckets.keys()), col2: list(buckets.values())}
        )
        return buckets_df

    @staticmethod
    def write_buckets_to_csv(
        buckets: Dict[str, List[int]], col1: str, col2: str, filename: str
    ):
        # Open the file in write mode
        with open(filename, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([col1, col2])

            for key, value in buckets.items():
                list_as_string = str(value).strip("[]")
                writer.writerow([key, list_as_string])

    @staticmethod
    def hamming_distance(str1: str, str2: str) -> int:
        if len(str1) != len(str2):
            raise ValueError("Strings must be of equal length")
        return sum(char1 != char2 for char1, char2 in zip(str1, str2))

    @staticmethod
    def get_text_idx(hamming_distances: List[int]) -> int:
        if len(hamming_distances) == 0:
            raise ValueError("No hamming distances found")
        return int(np.argmin(hamming_distances))


# def main():
#     # Load data
#     df = pl.read_csv(generated_data_file)
#     original_texts = df.get_column("original").to_numpy().flatten()
#     very_similar_texts = df.get_column("very_similar").to_numpy().flatten()

#     # Initialize LSH with the number of bits and seed
#     embedding_size = 3072  # Adjust based on your model's embedding size
#     lsh = LSH(nbits=nbits, embedding_size=embedding_size, seed=seed)

#     # Generate embeddings for both original and very similar texts
#     original_embeddings = [LSH.get_embedding(text, model) for text in original_texts]
#     very_similar_embeddings = [LSH.get_embedding(text, model) for text in very_similar_texts]

#     # Proceed with LSH operations (hashing, bucketing) on your embeddings
#     # For example, hashing original texts
#     original_hashed = [lsh.hash_vector(embedding) for embedding in original_embeddings]
#     very_similar_hashed = [lsh.hash_vector(embedding) for embedding in very_similar_embeddings]
#     # Bucket the hashes
#     original_buckets = LSH.bucket_hashes(original_hashed)
#     very_similar_buckets = LSH.bucket_hashes(very_similar_hashed)

#     print(original_buckets)
#     print(very_similar_buckets)
#     for i in range(original_hashed.__len__()):
#         for j in range(very_similar_hashed.__len__()):
#             print(i,j,lsh.hamming_distance(original_hashed[i], very_similar_hashed[j]))

# if __name__ == "__main__":
#     main()


#     # =============== B. Bucketing of a given text into available buckets===============
#     # search query
#     # ===== Type-1 =====
#     query = infos[0]  # try with the 1st one to verify the correctness
#     # query = infos[1]  # try with the 1st one to verify the correctness

#     # ===== Type-2 =====
#     # query = "I have bought many of the Vitality canned dog food products and have found them all to be of good quality. The product looks more like a stew than a processed meat and it smells good. My Labrador is finicky and she likes this product better than  most."  # changed the 1st review a bit
#     # query = 'Product reached marked as Jumbo Salted Peanuts...the peanuts were actually small sized unsalted. Not sure if this was a mistake or if the vendor wanted to indicate the product as "Jumbo".'  # changed the 2nd review a bit
#     # query = 'Great taffy at a better price.  There was a broad assortment of yummy taffy.  Delivery was super fast.  If your a taffy lover, this is a good chance.'  # changed the 5th review a bit

#     # ===== Type-3 =====
#     # query = "I've purchased numerous cans of the Vitality dog food line and have consistently found them to be of high quality. They resemble stew more than they do processed meat, and they have a more pleasant aroma. My picky Labrador prefers this brand over many others." # ai-generated the 1st review a bit
#     # query = 'Excellent value for delicious taffy. The selection offered a broad variety of delectable flavors. The shipping was impressively fast. For enthusiasts of taffy, this offer is a must-grab.'    # ai-generated the 5th review a bit
#     # query = "This dog food is highly nutritious and beneficial for digestive health. It's also suitable for young puppies. My dog consistently consumes the recommended portion at each meal."  # ai-generated the index-9 review a bit

#     # ===== Type-4 =====
#     # query = "I have lived out of the US for over 7 yrs now, and I so miss my Twizzlers!!  When I go back to visit or someone visits me, I always stock up.  All I can say is YUM!<br />Sell these in Mexico and you will have a faithful buyer, more often than I'm able to buy them right now."
#     # query = "Good oatmeal.  I like the apple cinnamon the best.  Though I wouldn't follow the directions on the package since it always comes out too soupy for my taste.  That could just be me since I like my oatmeal really thick to add some milk on top of."
#     # query = "I roast at home with a stove-top popcorn popper (but I do it outside, of course). These beans (Coffee Bean Direct Green Mexican Altura) seem to be well-suited for this method. The first and second cracks are distinct, and I've roasted the beans from medium to slightly dark with great results every time. The aroma is strong and persistent. The taste is smooth, velvety, yet lively."
#     hash_query = hash_vector(get_embedding(query), plane_norms)
#     # hash_query = hash_vector(get_embedding(query, model), plane_norms)
#     print(f"\nFor a given text: \"{query}\", it's computed hash is '{hash_query}'.")

#     # calculate the hamming distance between the query and each bucket
#     print("\nhamming distances b/w the query from each bucket key:")
#     hamming_distances = []
#     for hash_str in buckets.keys():
#         hamming_distances.append(hamming_distance(hash_query, hash_str))
#     print(hamming_distances)
#     if contains_zero(hamming_distances):
#         print(
#             "🙂 The given text falls into the bucket with its key having exact same hash"
#         )
#     else:
#         print(
#             "☹️ As no exact hash found, deliberately the closest bucket with min. hamming distance is selected here from left --> right."
#         )
#     # Get the index of the lowest one
#     min_index = np.argmin(hamming_distances)
#     print(
#         f"\nHence, the given text belongs to the index-{min_index} of bucket list, \ni.e. the bucket with key: '{list(buckets.keys())[min_index]}', value: [{list(buckets.values())[min_index]}]."
#     )


# if __name__ == "__main__":
#     main()
