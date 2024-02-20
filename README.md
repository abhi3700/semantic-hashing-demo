# Semantic Hashing Demo

## Description

A simple demo of semantic hashing of search queries. Also put the queries into buckets depending on the number of hyperplanes.

## Build

```sh
huak build
```

## Run

```sh
huak run main
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
