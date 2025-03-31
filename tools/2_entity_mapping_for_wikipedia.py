"""
Add two more fields to `output.jsonl` from `tools/bulk_inference.py`:
        1. labels_from_data (single_token from existing data)
        2. labels_from_llm (single_token from llama2 generation)
"""

import os
import sys

# Get the absolute path of the current directory
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(CUR_DIR, "..")

# From third-party libraries
import json
import pandas as pd
from jsonargparse import CLI
from tqdm import tqdm

# Add the root directory to the system path
sys.path.append(ROOT_DIR)

from tools.evaluator import (
    _clean_title,
    clean_text_generation,
    editdistance,
    ranks2metrics,
)

tqdm.pandas()


def parse_llm_generation_as_single_token(
    row, name2single, clean_name_pool, name_pool
):
    resp = row["completion"]

    # 1. Get the prediction list
    pred_list = get_pred_list(resp)

    # 2. Clean the pred text
    pred_list = [pred.lower() for pred in pred_list]

    # 3. Match existing movie names
    matched_single_tokens = []
    for pred in pred_list:
        # 3.1 Search the nearest movie name
        min_pos, min_dist = 0, float("inf")
        for i, clean_name in enumerate(clean_name_pool):
            dist = editdistance(pred, clean_name)
            if dist == 0:
                min_pos = i
                break
            if dist < min_dist:
                min_dist, min_pos = dist, i
        matched_single_token = name2single[name_pool[min_pos]]["single_token"]

        matched_single_tokens.append(matched_single_token)

    assert len(matched_single_tokens) == len(pred_list)
    return matched_single_tokens


def get_pred_list(resp):
    resp = resp.replace("The movie is ", "").replace("</s>", "")
    return clean_text_generation(resp)


def main(
    result_path: str = "reindex_step/data/wikipedia/output.jsonl",
    n: int = 0,
    N: int = 1,
    test: bool = False,
):
    assert ".jsonl" in result_path, (
        "Please provide the path to the result file in JSONL format."
    )

    # Load data
    name2single = json.load(open("data/item_entity_name_to_item_indices.json"))

    df = pd.read_json(result_path, lines=True)

    if N > 1:
        length = len(df)
        start = length * n // N
        end = length * (n + 1) // N
        df = df.iloc[start:end]

    # Set name pool for nearest search
    name_pool = list(name2single.keys())
    clean_name_pool = [
        _clean_title(name, is_from_dbpedia="dbpedia.org" in name).lower()
        for name in name_pool
    ]

    # Get the recommendation list
    df["labels_from_llm_as_single_token"] = df.progress_apply(
        lambda x: parse_llm_generation_as_single_token(
            x, name2single, clean_name_pool, name_pool
        ),
        axis=1,
    )

    # Save the result
    with open(
        f"{result_path.split('.jsonl')[0]}-mapping-{n}-{N}.jsonl",
        "w",
        encoding="utf-8",
    ) as f:
        df.to_json(f, orient="records", lines=True)

    # Test the result
    if test:

        def num_list2rank(gt_name, pred_names, MAX_RANK=10000):
            rank = MAX_RANK
            for i, pred_name in enumerate(pred_names):
                if pred_name == gt_name:
                    rank = i
                    break
            return rank

        ranks = []

        for _, row in df.iterrows():
            for gt_name in row["labels_from_data_as_single_token"]:
                ranks.append(
                    num_list2rank(
                        gt_name, row["labels_from_llm_as_single_token"]
                    )
                )

        metrics = ranks2metrics(ranks)
        print(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    CLI(main)
