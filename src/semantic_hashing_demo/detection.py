import polars as pl

def main():
    # load data from multiple CSV files in bulk
    df = pl.read_csv("output/test.csv").with_columns(pl.col('Hash').cast(pl.String))

if __name__ == "__main__":
    main()