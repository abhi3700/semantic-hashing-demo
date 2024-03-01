import polars as pl
from config import embedding_size, model, seed
from lsh import LSH
from utils import ensure_file_exists

dir_name = "output"
file_name = "paragraphs_processed.csv"


def main():
    # load data
    df = pl.read_csv("data/paragraphs.csv")
    source_texts = df.get_column("original").to_list()
    variants_texts = df.get_column("very_similar").to_list()

    # get embeddings for source and variants
    source_embeddings = LSH.get_embedding(source_texts, model)
    variant_embeddings = LSH.get_embedding(variants_texts, model)

    # Create DataFrame with embeddings of source and variants
    df2 = pl.DataFrame(
        {
            "Source": source_texts,
            "Variant": variants_texts,
            "Source Embedding": [
                str(embedding) for embedding in source_embeddings.tolist()
            ],
            "Variant Embedding": [
                str(embedding) for embedding in variant_embeddings.tolist()
            ],
        }
    )

    print("Saving Embeddings, LSH codes & HD matrix...\n")
    for nbits in [8, 16, 32, 64, 128]:
        # Create LSH instance
        lsh = LSH(nbits=nbits, seed=seed, embedding_size=embedding_size)

        hashes_source = lsh.hash_vector(source_embeddings)
        hashes_variant = lsh.hash_vector(variant_embeddings)

        # Add LSH hashes corresponding to the embeddings to the df2 DataFrame
        df2.insert_column(
            len(df2.columns), pl.Series(f"Source Hash {nbits}-bit", hashes_source)
        )
        df2.insert_column(
            len(df2.columns), pl.Series(f"Variant Hash {nbits}-bit", hashes_variant)
        )

        # Create matrix dataframe for each nbits
        df3 = pl.DataFrame(
            {"Sources": [f"Source-{i}" for i in range(len(hashes_source))]}
        )
        # calculate HD matrix
        for i, hash_variant in enumerate(hashes_variant):
            hamming_distances = [
                lsh.hamming_distance(hash_source, hash_variant)
                for hash_source in hashes_source
            ]

            # how to get nd array using numpy inserting column with a new dataframe
            df3.insert_column(
                i,
                pl.Series(f"Variant-{i}", hamming_distances),
            )

        df3.write_csv(f"{dir_name}/matrix_{nbits}.csv", separator=",")

        print(f"\tfor nbits = {nbits} ✅\n")

    # Ensure the file in desired path
    ensure_file_exists(dir_name, file_name)

    """ Save embeddings + LSH to CSV, linked with source sample """
    df2.write_csv(f"{dir_name}/{file_name}", separator=",")


if __name__ == "__main__":
    main()
