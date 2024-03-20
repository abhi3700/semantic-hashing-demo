# Semantic Hashing Demo

## Description

This analysis has 2 parts:

A. Analysis with given data
B. Analysis with Marvin-generated data

Details are covered [here](#run-analysis).

From an algorithmic standpoint, the Locality Sensitive Hashing (LSH) method is used here. This technique facilitates the creation of hyperplanes between different texts (e.g., food reviews), which are represented as embedding vectors. And in order to detect if a text is close to any given text, the hamming distance is calculated between their hashes looking at their binary bits if they differ. For instance `text_1` has LSH code of `1011`, and `text_2` has LSH code of `1100`, then the hamming distance is 3.
> The size of the embedding vector is 1536 for OpenAI's small embedding model, and 3072 for the large model.

Details on [Notion](https://www.notion.so/subspacelabs/Semantic-Hashing-Demo-38297cb7da594dcfb96393a3c491a936).

## Setup

Ensure the following steps are completed after cloning the repo:

1. Install `huak` project management tool (like `cargo` in Rust), following its [guide](https://github.com/cnpryer/huak/blob/master/docs/user_guide.md#installation).

Using pip (for any OS):

```sh
pip install git+https://github.com/cnpryer/huak@master#egg=huak
```

2. Download the data file as per [README](./data/README.md).
3. Build the cloned project:

```sh
cd ./semantic-hashing-demo
huak build
huak activate
pip install -r requirements.txt
```

> NOTE: After you are done, you may exit from the REPL-like terminal using typing `exit` or `ctrl+d`

## Run Analysis

This step is mainly to do the analysis as per objective stated [here](#description).

> Assuming you have exited from the REPL-like terminal before doing the analysis.

### Part-1: Analysis with given dataset

#### a. Generate buckets for Texts

Here, the dataset is based on [Amazon Food Reviews of size = 1000](./data/fine_food_reviews_1k.csv). So, each food review becomes the text that has to be bucketed based on their similarities. Now, the food reviews are grouped into buckets based on their semantic hash values. The semantic hash is determined based on [LSH approach](https://www.pinecone.io/learn/series/faiss/locality-sensitive-hashing-random-projection/). For more details, refer [this PR](https://github.com/abhi3700/semantic-hashing-demo/pull/1) as initial draft, which is then refactored (as per OOP) in [PR #2](https://github.com/abhi3700/semantic-hashing-demo/pull/2).

In this analysis, we have gone ahead with 8, 16, 32, 64, 128 hyperplanes. The idea is to classify given texts to the maximum extent (buckets) possible so that we can identify query texts that are fed into this model and see the desired results.

So, in this step, we thought of data pre-processing i.e. fetching the data from the dataset and then grouping them into buckets based on their semantic hash values. We are saving the details in these files:

- [preprocessed_data.csv](./output/preprocessed_data.csv): Save the texts along with their embedding values, as well as their semantic hash values for all 5 hyperplanes.
- [buckets_8bit.csv](./output/buckets_8bit.csv): Save the buckets containing text indices for 8 hyperplanes.
- [buckets_16bit.csv](./output/buckets_16bit.csv): Save the buckets containing text indices for 16 hyperplanes.
- [buckets_32bit.csv](./output/buckets_32bit.csv): Save the buckets containing text indices for 32 hyperplanes.
- [buckets_64bit.csv](./output/buckets_64bit.csv): Save the buckets containing text indices for 64 hyperplanes.
- [buckets_128bit.csv](./output/buckets_128bit.csv): Save the buckets containing text indices for 128 hyperplanes.

> Now, make sure you are at the project's root directory containing `pyproject.toml` file.

For data pre-processing, run this:

```sh
# pre-process input data to buckets
huak run preprocess
```

#### b. Detect multiple text types

This step is to identify multiple text types (similar, semantically similar, unrelated) in the dataset. Suppose, in order to detect if a text of one of the types is falling into the expected bucket i.e. the one that contains the original text or not.

So, how can we generate the query text samples? <br/>
In this step, we would be manually modifying the original text to generate the query text of required types for detection test here.

Although, we found a project/library - [Marvin](https://www.askmarvin.ai/) that provides `marvin` python package as well to write some script around generating datasets of desired types. But as of now, I haven't been able to figure out as to how can a given/original text (in the dataset) be modified as per desired type. For instance, if we need a slightly modified (replaced few words with its synonym) text, then we can modify the 1st review from:

```text
I have bought several of the Vitality canned dog food products and have found them all to be of good quality. The product looks more like a stew than a processed meat and it smells better. My Labrador is finicky and she appreciates this product better than  most.
```

to:

```text
I have bought many of the Vitality canned dog food products and have found them all to be of good quality. The product looks more like a stew than a processed meat and it smells good. My Labrador is finicky and she likes this product better than  most.
```

Similarly for other types - semantically similar (using LLMs) and unrelated.

> Now, make sure you are at the project's root directory containing `pyproject.toml` file.

Now, if we are ready with the text samples, let's run this for a single query text:

```sh
# detect similar text
huak run detect
```

> TODO: We can modify [script](src/semantic_hashing_demo/detection.py) to accomodate detecting query texts in bulk.

### Part-2: Analysis with Marvin-generated data

This part is to perform analysis with Marvin-generated data.

The analysis could be run via the commands in the sub-sections below or directly in the [notebook](./main.ipynb).

#### a. Generate data

In order to run a dataset of 2 columns: **text** and **similar text** with 20 counts, run this:

```sh
# generate texts (source, variant)
huak run generate
```

> We call original text as **source** and similar text as **variant**.
>
> Marvin allows you to define the [instruction](https://github.com/abhi3700/semantic-hashing-demo/blob/716618b38875dd93680ba8880d5a9c1d9bc9c901/src/semantic_hashing_demo/generate_data.py#L21) and [variants' count](https://github.com/abhi3700/semantic-hashing-demo/blob/716618b38875dd93680ba8880d5a9c1d9bc9c901/src/semantic_hashing_demo/generate_data.py#L19) for a custom defined [object](https://github.com/abhi3700/semantic-hashing-demo/blob/716618b38875dd93680ba8880d5a9c1d9bc9c901/src/semantic_hashing_demo/generate_data.py#L8-L13), based on the  the requirement.

#### b. Relationship analysis with Marvin generated data

In order to find the semantic relation between the source and variant texts, run this:

```sh
# process generated texts (source, variant)
huak run post_generate
```

This would generate heatmap plots showcasing the relationship between each source with all variants as matrices (where each cell has hamming distance value) when the hyperplanes no. is increased from 8 to 128. Based on the plots, it has been observed that with increase in the number of hyperplanes, the relationship between the texts with their similar text counterparts becomes crystal clear (depicted with blue colored diagonal line).

## Format

```sh
huak fmt
```

## Lint

```sh
huak lint
```

## Algorithm testing [Archived]

> This is part of [PR #1](https://github.com/abhi3700/semantic-hashing-demo/pull/1).

Refer this [README](./tests/README.md) for more details.
