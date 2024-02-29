from concurrent.futures import ThreadPoolExecutor, as_completed

import marvin
import polars as pl
from pydantic import BaseModel, Field


class ParagraphData(BaseModel):
    original: str = Field(description="The original paragraph")
    very_similar: str = Field(
        description="A paragraph that is almost identical to the original paragraph with only a couple of words changed"
    )
    # semantically_similar:str


# data = []
# for _ in range(5):
#     data.extend(marvin.generate(n=3, target=ParagraphData, instructions="generate paragraphs for comparison testing. the paragraphs should be almost identical, with only a few words changed. each paragraph should be at least 100 words long."))
#     print(data)
# data_dicts = [d.dict() for d in data]
# df = pl.DataFrame(data_dicts)
# df.write_csv("./data/paragraphs.csv")


def generate_data():
    print("generating data")
    new_data = marvin.generate(
        n=2,
        target=ParagraphData,
        instructions="generate paragraphs for comparison testing. the paragraphs should be almost identical, with only a few words changed. each paragraph should be at least 100 words long.",
    )
    return new_data


data = []

# Number of parallel calls you want to make
num_parallel_calls = 10

# Use ThreadPoolExecutor to execute calls in parallel
with ThreadPoolExecutor(max_workers=num_parallel_calls) as executor:
    # Submit all your generate calls to the executor
    future_to_generate = {
        executor.submit(generate_data) for _ in range(num_parallel_calls)
    }

    # Collect the results as they are completed
    for future in as_completed(future_to_generate):
        try:
            data.extend(future.result())
        except Exception as exc:
            print(f"Generated an exception: {exc}")

# Print the data collected
# for item in data:
#     print(item)

data_dicts = [d.dict() for d in data]
df = pl.DataFrame(data_dicts)
df.write_csv("./data/paragraphs.csv")
