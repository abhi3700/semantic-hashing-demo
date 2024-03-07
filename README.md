# Semantic Hashing Demo

## Description

Initially, the objective is to identify relationships among various types of texts - those that are similar, those that are semantically similar (generated using Large Language Models, or LLMs) and those that are completely unrelated. For this purpose, the "Amazon Food Reviews" dataset from Kaggle is utilized in this notebook.

Subsequently, texts are generated using [Marvin](https://www.askmarvin.ai/) through their `marvin` Python package. The dataset includes paragraph-sized texts and their semantically similar counterparts. Correspondingly, matrices are created and visualized as heatmap plots to demonstrate the semantic relationships, with an increasing number of hyperplanes.

From an algorithmic standpoint, the Locality Sensitive Hashing (LSH) method is employed. This technique facilitates the creation of hyperplanes between different texts (e.g., food reviews), which are represented as embedding vectors.
> The size of the embedding vector is 1536 for OpenAI's small embedding model, and 3072 for the large model.

Subsequently, for any given text, a semantic hash is computed. This process involves converting a large embedding vector (a numerical representation of text) into a few bits (representing the number of hyperplanes), akin to a hash code. For example, the phrase "The food was very delicious" could be represented as `1011` in a system with 4 hyperplanes. Finally, texts with identical hashes are grouped into multiple buckets.

Details on [Notion](https://www.notion.so/subspacelabs/Semantic-Hashing-Demo-38297cb7da594dcfb96393a3c491a936).

## Install

Ensure these:

1. `huak` is installed following the [guide](https://github.com/cnpryer/huak/blob/master/docs/user_guide.md#installation).

```sh
pip install git+https://github.com/cnpryer/huak@master#egg=huak
```

2. Download the data file as per [README](./data/README.md).
3. Add the dependencies:

```sh
huak build
huak activate
pip install -r requirements.txt

# optional
> exit or ctrl+d
```

## Run

```sh
# pre-process input data to buckets
huak run preprocess

# detect similar text
huak run detect

# generate texts (source, variant)
huak run generate

# process generated texts (source, variant)
huak run post_generate
```

## Format

```sh
huak fmt
```

## Lint

```sh
huak lint
```

## Algorithm testing

Refer this [README](./tests/README.md) for more details.
