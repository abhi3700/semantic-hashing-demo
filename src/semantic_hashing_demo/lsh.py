import csv
from typing import Dict, List

import numpy as np
import polars as pl

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
        buckets: Dict[str, List[int]], col1: str, col2: str, file_path: str
    ):
        # Open the file in write mode
        with open(file_path, "w", newline="") as file:
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
