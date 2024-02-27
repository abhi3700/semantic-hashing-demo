""" 
The end goal is to detect AI-generated text.
"""
import polars as pl
from config import preprocessed_data_file


def main():
    # Specify data types for multiple columns to be read as strings
    dtype_spec = {
        "Hash 8-bit": str,
        "Hash 16-bit": str,
        "Hash 32-bit": str,
        "Hash 64-bit": str,
        "Hash 128-bit": str,
    }

    # pull data into polars dataframe
    df = pl.read_csv(preprocessed_data_file, dtypes=dtype_spec)
    print(df.head())


if __name__ == "__main__":
    main()
