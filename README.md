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

# pre-processing data
huak run preproc

# detect AI-generated data
huak run detect
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
