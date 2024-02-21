# Algorithm testing

> No. of samples & hyperplanes are set such that there are adequate samples put to respective buckets. For instance, if there are 100 samples and 4 hyperplanes, then there are 16 (2^4) buckets. So, there are at least 6 samples in each bucket.

> Here, food reviews (pulled from [kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)) are used as the text list and one of the review is used to test the built model.

```txt
LEGEND
✅: Pass i.e. falls into expected bucket.
❌: Fail i.e. falls into different bucket than expected.
```

## Tests

Tests are done in different modes keeping the semantic meaning same.

### Type-1: Exact text

Parse a exact query text taken out of the text list & check if the text falls into the pre-categorized bucket.

Below are the results with openai small embedding model only.

All test passed ✅ with any exact text as trained.

### Type-2: Slightly Modified text

Parse a slightly modified text taken out of the text list & check if the text falls into the pre-categorized bucket.

Below are the results with openai small embedding model only.

#### text-0

[run-1](./20_4_2a1.txt), [run-2](./20_4_2a2.txt) ✅

#### text-1

[run-1](./20_4_2b1.txt), [run-2](./20_4_2b2.txt) ✅

#### text-4

[run-1](./20_4_2c1.txt). No need to repeat, as same result found. It fails ❌ as it falls into a different bucket than bucket with key: `1110`. Given its 4 bits, the hashes are different by 1 bit.

```txt
original text: 1110
modified text: 1100
```

### Type-3: AI-generated text

Parse an AI-generated version of a query text taken out of the text list & check if the text falls into the pre-categorized bucket.

> Used the following prompt in GPT-4 for this.

```txt
Create a text with same semantic meaning as ".....".
```

Below are the results with openai small/large embedding models.

#### text-0

- <u>`{bucketing: small, query: small}`</u>: [run-1](./40_4_3a.txt). It fails ❌ as it falls into a different bucket than a bucket containing 'text-0'. Also, the hash is slightly different from the original text.

  ```txt
  original text hash: 1001
  generated text hash: 1000
  ```

  It seems like we need to switch to a better embedding model with more embedding values like `text-embedding-3-large` from available [OpenAI embedding models](https://platform.openai.com/docs/guides/embeddings/embedding-models).

- <u>`{bucketing: large, query: large}`</u>: When large model used for both, the hashes were different. [run-3](./40_4_3a3.txt) ✅

- <u>`{bucketing: large, query: small}`</u> [run-2](./40_4_3a2.txt) ❌

#### text-4

- <u>`{bucketing: small, query: small}`</u>: [run-1](./40_4_3b.txt). It fails ❌ as it falls into a different bucket than a bucket containing 'text-4'. Also, the hash is very different from the original text.

```txt
original text hash: 1110
generated text hash: 0010
```

- <u>`{bucketing: large, query: large}`</u>: When large model used for both, the hashes were different. [run-2](./40_4_3b2.txt) ❌

```txt
original text hash: 0010
generated text hash: 1010
```

- <u>`{bucketing: large, query: small}`</u> [run-2](./40_4_3b3.txt) ✅

#### text-9

- <u>`{bucketing: small, query: small}`</u>: [run-1](./40_4_3c.txt). It fails ❌ as it falls into a different bucket than a bucket containing 'text-9'. Also, the hash is very different from the original text.

```txt
original text hash: 1000
generated text hash: 1100
```

- <u>`{bucketing: large, query: large}`</u>: [run-2](./40_4_3c2.txt) ❌. The hashes differs by 1 bit.

```txt
original text hash: 0011
generated text hash: 0001
```

- <u>`{bucketing: large, query: small}`</u>: [run-3](./40_4_3c3.txt) ❌. The hashes are completely different.

```txt
original text hash: 0011
generated text hash: 1100
```

## Conclusion

Had to also consider openai large embedding model as well. In some cases, tests passed as in the query text fell into the expected bucket.
At this point it’s a bit ambiguous to jump to a conclusion. There are 2 main observations as of now:

1. In cases where the semantic hash is different for the original & query texts, embedding model needs to be replaced.
2. In cases where the semantic hash is same for the original & query texts, hamming distance vector is to blamed for. For instance, with [2, 1, 1, 3, 4] hamming distance vector, the traversal if done from left, the query text is supposed to go into bucket at index-1 as it is the shortest distance so far. But, the original text is inside the bucket at index-2.
