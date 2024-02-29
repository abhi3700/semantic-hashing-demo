import polars as pl
from config import embedding_size, model, seed
from lsh import LSH

# slightly modified 1st review from the dataset
query = "I have bought many of the Vitality canned dog food products and have found them all to be of good quality. The product looks more like a stew than a processed meat and it smells good. My Labrador is finicky and she likes this product better than  most."


def main():
    for nbits in [8, 16, 32, 64, 128]:
        print(f"\n=====For nbits = {nbits}======\n")

        # instantiate LSH
        lsh = LSH(nbits=nbits, embedding_size=embedding_size, seed=seed)

        # get hash of a query text
        query_hash = lsh.hash_vector(lsh.get_embedding([query], model))[0]
        print(
            f"\nFor a given text: \n\"{query}\", \nit's computed hash is '{query_hash}'."
        )

        # load data
        df = pl.read_csv(f"output/buckets_{nbits}bit.csv", dtypes={"Text Hash": str})
        bucket_hashes = df.get_column("Text Hash").to_numpy().flatten()
        bucket_indices = df.get_column("Text Indices").to_numpy().flatten()

        # get hamming distances between the query and each bucket key
        hamming_distances = [
            lsh.hamming_distance(query_hash, hash_str) for hash_str in bucket_hashes
        ]

        if 0 in hamming_distances:
            print("🎉 Falls into exact bucket!")
        else:
            print("😟 Falls into closest bucket, when traversed from left --> right.")
        print(
            f"The bucket contains texts at indices: {bucket_indices[lsh.get_text_idx(hamming_distances)]}."
        )


if __name__ == "__main__":
    main()
