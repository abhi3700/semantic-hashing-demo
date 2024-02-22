# Algorithm testing

> No. of samples & hyperplanes are set such that there are adequate samples put to respective buckets. For instance, if there are 100 samples and 4 hyperplanes, then there are 16 (2^4) buckets. So, there are at least 6 samples in each bucket.

> Here, food reviews (pulled from [kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)) are used as the text list and one of the review is used to test the built model.

```txt
LEGEND
✅: Pass
❌: Fail

Should return result as expected depending on the case.
```

`run-1-{40-4}`: means 40 samples, 4 hyperplanes.

## Tests

Tests are done in different modes keeping the semantic meaning same.

### Type-1: Exact text

<u>Objective</u>: Parse a exact query text taken out of the text list & check if the text falls into the pre-categorized bucket.

Below are the results with openai small embedding model only.

All test passed ✅ with any exact text as trained.

### Type-2: Slightly Modified text

<u>Objective</u>: Parse a slightly modified text taken out of the text list & check if the text falls into the pre-categorized bucket.

Below are the results with openai small embedding model only.

#### text-0

- [run-1-{20-4}](./20_4_2a1.txt): As **one of the hamming distance is zero**, the query text falls into a bucket that has "text-0". Max. hamming distance is 3.
- [run-2-{20-8}](./20_8_2a2.txt): As **one of the hamming distance is zero**, the query text falls into a bucket, but that doesn't have "text-0", instead "text-3". Max. hamming distance is 5.

#### text-1

- [run-1-{20-4}](./20_4_2b1.txt): As **one of the hamming distance is zero**, the query text falls into a bucket that has "text-1". Max. hamming distance is 3.
- [run-2-{20-8}](./20_8_2b2.txt): As **one of the hamming distance is zero**, the query text falls into a bucket that has "text-1". Max. hamming distance is 5.

#### text-4

- [run-1-{20-4}](./20_4_2c1.txt): As **one of the hamming distance is zero**, the query text falls into a bucket, but that doesn't have "text-4", instead "[1, 6, 7, 13]". Max. hamming distance is 3.
- [run-2-{20-8}](./20_8_2c2.txt): As **none of the hamming distance is zero**, the query text falls into the closest bucket that has "text-4". Max. hamming distance is 6.

### Type-3: Similar ~~(AI-generated)~~ text

> Prefer to call it "Similar" than "AI generated".

<u>Objective</u>: Parse an AI-generated version of a query text taken out of the text list & check if the text falls into the pre-categorized bucket. Here, we are aiming to catch the similar ~~(AI-generated)~~ content that has the same meaning.

> Used the following prompt in GPT-4 for this.

```txt
Create a text with same semantic meaning as ".....".
```

Below are the results with openai small/large embedding models.

#### text-0

- <u>`{bucketing: small, query: small}`</u>:
  - [run-1](./40_4_3a.txt): As **one of the hamming distance is zero**, the query text falls into a bucket, but that doesn't have the original "text-0", instead "[9, 28, 30, 33]". In this case, how to raise the red flag that we found a similar content?❓ 🤔 Max. hamming distance is 3.
  - [run-2-{20-8}](./20_8_3a4.txt): As **none of the hamming distance is zero**, the query text is deliberately put into the closest bucket that doesn't have the original "text-0", instead "[1, 13]". Hence, we have caught the similar content bcoz as no HD value is zero. Max. hamming distance is 5.

<!--
  It seems like we need to switch to a better embedding model with more embedding values like `text-embedding-3-large` from available [OpenAI embedding models](https://platform.openai.com/docs/guides/embeddings/embedding-models).

- <u>`{bucketing: large, query: large}`</u>: When large model used for both, the hashes were different. [run-3](./40_4_3a3.txt) ✅

- <u>`{bucketing: large, query: small}`</u> [run-2](./40_4_3a2.txt) ❌ -->

#### text-4

- <u>`{bucketing: small, query: small}`</u>:
  - [run-1-{40-4}](./40_4_3b.txt): As **one of the hamming distance values is zero**, the query text falls into a bucket, but that doesn't have "text-4", instead "text-18". Max. hamming distance is 4.
  - [run-2-{20-8}](./20_8_3b2.txt):  As **none of the hamming distance is zero**, the query text is deliberately put into the closest bucket that has "text-4". Hence, we have caught the similar content bcoz as no HD value is zero. Max. hamming distance is 7.
  
<!-- - <u>`{bucketing: large, query: large}`</u>: When large model used for both, the hashes were different. [run-2](./40_4_3b2.txt) ❌

```txt
original text hash: 0010
generated text hash: 1010
```

- <u>`{bucketing: large, query: small}`</u> [run-2](./40_4_3b3.txt) ✅ -->

#### text-9

- <u>`{bucketing: small, query: small}`</u>:
  - [run-1-{40-4}](./40_4_3c.txt): As **one of the hamming distance values is zero**, the query text falls into a bucket, but that doesn't have "text-9", instead "[1, 6, 7, 13, 22, 23, 24, 26, 33, 34]". Max. hamming distance is 3.
  - [run-2-{20-8}](./20_8_3c4.txt). As **none of the hamming distance is zero**, the query text is deliberately put into the closest bucket, but that doesn't have "text-9", instead "[1, 13]". Max. hamming distance is 5. Hence, we have caught the similar content bcoz as no HD value is zero. Max. hamming distance is 7.

<!-- - <u>`{bucketing: large, query: large}`</u>: [run-2](./40_4_3c2.txt) ❌. The hashes differs by 1 bit.

```txt
original text hash: 0011
generated text hash: 0001
```

- <u>`{bucketing: large, query: small}`</u>: [run-3](./40_4_3c3.txt) ❌. The hashes are completely different.

```txt
original text hash: 0011
generated text hash: 1100
``` -->

[🚧 WIP] <u>Conclusion</u>: For similar (aka AI-generated) query text, we are able to catch the similar content, testing with higher dimensionality (aka hyperplanes). This can be confirmed looking at the hamming distance vector not having any 0 value inside.

### Type-4: Completely unrelated text

Unrelated to the given samples. For instance, if 20 samples used then give a text i.e. not related to any one of them.

TODO: Need to test this.

## Conclusion

Had to also consider openai large embedding model as well. In some cases, tests passed as in the query text fell into the expected bucket.
At this point it’s a bit ambiguous to jump to a conclusion. There are 2 main observations as of now:

1. In cases where the semantic hash is different for the original & query texts, embedding model needs to be replaced.
2. In cases where the semantic hash is same for the original & query texts, hamming distance vector is to blamed for. For instance, with [2, 1, 1, 3, 4] hamming distance vector, the traversal if done from left, the query text is supposed to go into bucket at index-1 as it is the shortest distance so far. But, the original text is inside the bucket at index-2.
