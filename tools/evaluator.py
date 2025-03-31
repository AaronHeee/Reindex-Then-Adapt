"""Evaluator for the conversational recommendation dataset."""

import re
import os
import numpy as np
import pandas as pd

from editdistance import eval as editdistance


def ranks2metrics(ranks, Ks=[1, 5, 10]):
    """From ranks to metrics:

    Args:
        ranks: list of ranks, starting from 0
        Ks: list of Ks

    Returns:
        metrics: dict of metrics, with standard errors
    """

    metrics = dict()
    for K in Ks:
        # Recall@K
        Recall_at_K = [r < K for r in ranks]
        metrics[f"Recall@{K}"] = np.mean(Recall_at_K)
        metrics[f"Recall@{K}_SE"] = np.std(Recall_at_K) / np.sqrt(len(ranks))

        # NDCG@K
        NDCG_at_K = [1 / np.log2(r + 2) if r < K else 0 for r in ranks]
        metrics[f"NDCG@{K}"] = np.mean(NDCG_at_K)
        metrics[f"NDCG@{K}_SE"] = np.std(NDCG_at_K) / np.sqrt(len(ranks))

        # MRR@K
        MRR_at_K = [1 / (r + 1) if r < K else 0 for r in ranks]
        metrics["MRR"] = np.mean(MRR_at_K)
        metrics["MRR_SE"] = np.std(MRR_at_K) / np.sqrt(len(ranks))

    return metrics


def save_pred_file(save_path, conv_ids, turn_ids, item_ids, top_20, pred_ranks):
    """Save the prediction file.

    Args:
        save_dir: str, directory to save the prediction file
        conv_ids: list of str, conversation ids
        turn_ids: list of str, turn ids
        item_ids: list of str, item ids
        top_20: list of list of str, top 20 item ids
        pred_ranks: list of int, predicted ranks
    """

    assert (
        len(conv_ids)
        == len(turn_ids)
        == len(item_ids)
        == len(top_20)
        == len(pred_ranks)
    )

    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)

    df = pd.DataFrame(
        {
            "conv_id": conv_ids,
            "turn_id": turn_ids,
            "item_id": item_ids,
            "pred_top_20": top_20,
            "pred_rank": pred_ranks,
        }
    )

    df.to_csv(save_path, index=False)
    print(f"Saved to {save_path}")


def _del_parentheses(text):
    pattern = r"\([^()]*\)"
    return re.sub(pattern, "", text)


def _del_space(text):
    pattern = r"\s+"
    return re.sub(pattern, " ", text).strip()


def _del_numbering(text):
    pattern = r"^(?:\d+[\.\)、]?\s*[\-\—\–]?\s*)?"
    return re.sub(pattern, "", text)


def _del_punctuation(text):
    pattern = r"[^\w\s]"
    return re.sub(pattern, "", text)


def _clean_dbpedia_title(title):
    return title.split("/")[-1].replace("_", " ").replace(">", "").strip()


def _clean_title(title, is_from_dbpedia=False):
    if is_from_dbpedia:
        title = _clean_dbpedia_title(title)
    return _del_space(
        _del_punctuation(_del_parentheses(_del_numbering(title.strip())))
    )


def clean_text_generation(text):
    """Clean the text generation output.

    Args:
        text: str, text generation output

    Returns:
        text: str, cleaned text generation output
    """

    # we assume the recommendation list starts from "1."
    try:
        _, text = text.split("1.", maxsplit=1)
    except Exception as e:
        text = text.replace(",", "\n")

    # we further assume that each item is separated by a "\n"
    item_list = text.split("\n")

    return [_clean_title(item.strip()) for item in item_list]


def _equal(a, b):
    return editdistance(a, b) <= 2


def list2rank(gt_name, pred_names, MAX_RANK=10000, gt_from_dbpedia=False):
    """Get the rank of the ground truth item in the prediction list.

    Args:
        gt_name: str, ground truth item name
        pred_names: list of str, prediction item names

    Returns:
        rank: int, rank of the ground truth item in the prediction list
    """

    # lower the strings
    pred_names = [pred_name.lower() for pred_name in pred_names]

    gt_name = _clean_title(gt_name, gt_from_dbpedia).lower()

    # if the ground truth item is not in the prediction list, we return MAX_RANK
    rank = MAX_RANK
    for i, pred_name in enumerate(pred_names):
        if _equal(pred_name, gt_name):
            rank = i
            break
    return rank
