""" 
This module contains the input data for the semantic hashing demo.
"""


# no. of hyperplanes
nbits = 8


# data file
# NOTE: This data file has around 570k text reviews (of types: single line, paragraph).
# So, parse accordingly depending on the computational resources for bucketing.
data_file = "./data/fine_food_reviews_1k.csv"
preprocessed_data_file = "./output/preprocessed_data.csv"
generated_data_file = "./data/paragraphs.csv"

# no. of text samples
n = 20

# seed for hyperplane generation
seed = 2254  # subspace address format prefix

# embedding model
model = "text-embedding-3-small"
# model = "text-embedding-3-large"

# embedding size
embedding_size = int(1536)  # for small
# embedding_size = int(3072)   # for large
