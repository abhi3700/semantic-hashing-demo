from pathlib import Path
import polars as pl
from lsh import LSH
from config import embedding_size, model, seed
from utils import ensure_file_exists

OUTPUT_DIR_NAME = "output"
output_dir = Path(OUTPUT_DIR_NAME)


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
            "Source Embedding": [str(embedding) for embedding in source_embeddings.tolist()],
            "Variant Embedding": [str(embedding) for embedding in variant_embeddings.tolist()],
        }
    )
    
    for nbits in [8, 16, 32, 64, 128]:
        # Create LSH instance
        lsh = LSH(nbits=nbits, seed=seed, embedding_size=embedding_size)
        
        hashes_source = lsh.hash_vector(source_embeddings)
        hashes_variant = lsh.hash_vector(variant_embeddings)
        
        # Add LSH hashes corresponding to the embeddings to the df2 DataFrame
        df2.insert_column(len(df2.columns), pl.Series(f"Source Hash {nbits}-bit", hashes_source))
        df2.insert_column(len(df2.columns), pl.Series(f"Variant Hash {nbits}-bit", hashes_variant))

    
    # Define the path for the directory and the file
    file_path = output_dir / "paragraphs_processed.csv"
    ensure_file_exists(output_dir, file_path)
    
    """ Save embeddings + LSH to CSV, linked with source sample """
    df2.write_csv(file_path, separator=",")
    

if __name__ == "__main__":
    main()