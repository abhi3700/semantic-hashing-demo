""" 
This module contains the input data for the semantic hashing demo.
"""


# no. of hyperplanes
nbits = 4


# data file
# NOTE: This data file has around 570k text reviews (single line vs paragraph).
# So, parse accordingly depending on the computational resources for bucketing.
data_file = "./data/fine_food_reviews_1k.csv"

# no. of text samples
n = 10

# seed for hyperplane generation
seed = 'subspace'