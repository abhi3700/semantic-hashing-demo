# Semantic Hashing Demo

## Description

A simple demo of semantic hashing of search queries. Also put the queries into buckets depending on the number of hyperplanes.

Details on [Notion](https://www.notion.so/subspacelabs/Semantic-Hashing-Demo-38297cb7da594dcfb96393a3c491a936).

## Install

Ensure these:

1. `huak` is installed following the [guide](https://github.com/cnpryer/huak/blob/master/docs/user_guide.md#installation).
2. Download the data file as per [README](./data/README.md).

## Build

```sh
huak build
```

## Run

```sh
# demo-1
huak run main

# demo-2
huak run main2
```

## Format

```sh
huak fmt
```

## Lint

```sh
huak lint
```

## Benchmark

Tested on my machine: **Apple M1 Max chip, 32GB RAM, 10 cores**.

| Text inputs | no. of hyperplanes | Execution Time |
| :---------: | :----------------: | :------------: |
|     100     |         4          |    1m 5.69s    |
|     150     |         4          |   3m 38.33s    |
|     200     |         4          |   5m 18.48s    |
|     500     |         4          |   15m 39.88s   |

## Algorithm testing

Refer this [README](./tests/README.md) for more details.
