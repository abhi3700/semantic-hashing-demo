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

[run-1](./20_4_2a1.txt), [run-2](./20_4_2a2.txt) ✅

#### text-1

[run-1](./20_4_2b1.txt), [run-2](./20_4_2b2.txt) ✅

#### text-4

- [run-1-{20-4}](./20_4_2c1.txt): It fails ❌ as it falls into a different bucket than bucket with key: `1110`. Given its 4 bits, the hashes are different by 1 bit.
- [run-2](./20_8_2c2.txt): Although none of the hamming distance values is zero, the query text is enforced to fall into the bucket that has original "text-4". This means the search query has a different semantic meaning than the original text based on the fact that the hamming distance values are not zero.

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
  - [run-1](./40_4_3a.txt): As **one of the hamming distance is zero**, the query text falls into a bucket that doesn't have the original "text-0". In this case, how to raise the red flag that we found a similar content?❓ 🤔
  - [run-2-{20-8}](./20_8_3a4.txt): As **none of the hamming distance is zero**, the query text is deliberately put into the closest bucket that doesn't have the original "text-0". Hence, we have caught the similar content.

<!--
  It seems like we need to switch to a better embedding model with more embedding values like `text-embedding-3-large` from available [OpenAI embedding models](https://platform.openai.com/docs/guides/embeddings/embedding-models).

- <u>`{bucketing: large, query: large}`</u>: When large model used for both, the hashes were different. [run-3](./40_4_3a3.txt) ✅

- <u>`{bucketing: large, query: small}`</u> [run-2](./40_4_3a2.txt) ❌ -->

#### text-4

- <u>`{bucketing: small, query: small}`</u>:
  - [run-1-{40-4}](./40_4_3b.txt): Here, **one of the hamming distance values is zero**. Hence, the query text intents to falls into the bucket which has 19th review.
  - [run-2-{20-8}](./20_8_3b2.txt): Although search query falls into the bucket that has original "text-4", but as **none of the hamming distance values is zero**, so we have caught the similar content.
  
<!-- - <u>`{bucketing: large, query: large}`</u>: When large model used for both, the hashes were different. [run-2](./40_4_3b2.txt) ❌

```txt
original text hash: 0010
generated text hash: 1010
```

- <u>`{bucketing: large, query: small}`</u> [run-2](./40_4_3b3.txt) ✅ -->

#### text-9

- <u>`{bucketing: small, query: small}`</u>:
  - [run-1-{40-4}](./40_4_3c.txt): Here, search query prefers to fall into the bucket that doesn't have 10th review as the hamming distance is 0 for that bucket key. Hence, not able to catch the similar content.
  - [run-2-{20-8}](./20_8_3c4.txt). As expected, it ideally doesn't go into any bucket as none of the hamming distance values is zero. Hence, we have caught the similar content.

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

### Type-4: Completely unrelated text

Unrelated to the given samples. For instance, if 20 samples used then give a text i.e. not related to any one of them.

TODO: Need to test this.

## Conclusion

Had to also consider openai large embedding model as well. In some cases, tests passed as in the query text fell into the expected bucket.
At this point it’s a bit ambiguous to jump to a conclusion. There are 2 main observations as of now:

1. In cases where the semantic hash is different for the original & query texts, embedding model needs to be replaced.
2. In cases where the semantic hash is same for the original & query texts, hamming distance vector is to blamed for. For instance, with [2, 1, 1, 3, 4] hamming distance vector, the traversal if done from left, the query text is supposed to go into bucket at index-1 as it is the shortest distance so far. But, the original text is inside the bucket at index-2.
