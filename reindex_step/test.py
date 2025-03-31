"""
Example:
    CUDA_VISIBLE_DEVICES=4 python reindex_step/test.py --load_pretrained_weights logs/inspired_redial_redditv1.5/rnn/version_1/checkpoints/best.ckpt
"""

# Define Dependency

import os
import sys
import json
from collections import defaultdict

# Get the absolute path of the current directory
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(CUR_DIR, "..")

# Add the root directory to the system path
sys.path.append(ROOT_DIR)

from aggregators import AGGREGATORS
from tools.evaluator import ranks2metrics

# From third-party libraries
from jsonargparse import CLI, Namespace

import torch
from torch import nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from tqdm import tqdm
import numpy as np


## Define Dataset


class Data(object):
    def __init__(self, args):
        """Build data according to args.data_names

        Attributes:
            samples_with_item_targets: test / valid / test samples with item targets
                - Example: x['test']['embedding'],  x['test']['single_token'], x['test']['conv_id'], x['test']['turn_id']

            embed_size: int, embedding size
            num_items: int, number of items

            single2llm: dict, mapping from single token to llm tokens
            data2single: dict, mapping from data to the list of single token within the data
        """
        self.args = args

        print(
            "Building the `samples_with_item_targets` used for training, validation, and testing..."
        )

        self.samples_with_item_targets = {}
        for data_name in args.data_names:
            for split in ["valid", "test"]:
                path = os.path.join(
                    ROOT_DIR,
                    "reindex_step/data",
                    data_name,
                    split,
                    "output-encoding.jsonl",
                )
                for line in tqdm(
                    open(path, "r"), desc=f"loading {data_name} {split}:"
                ):
                    sample = json.loads(line)
                    MAX_LEN = (
                        5
                        if args.label == "labels_from_llm_as_single_token"
                        else 100
                    )
                    for sing_token in sample[args.label][:MAX_LEN]:
                        self.samples_with_item_targets.setdefault(
                            split, defaultdict(list)
                        )
                        self.samples_with_item_targets[split][
                            "embedding"
                        ].append(sample["encoding"])
                        self.samples_with_item_targets[split][
                            "single_token"
                        ].append(sing_token)
                        self.samples_with_item_targets[split]["dataset"].append(
                            data_name
                        )
                        self.samples_with_item_targets[split]["conv_id"].append(
                            sample["conv_id"]
                        )
                        self.samples_with_item_targets[split]["turn_id"].append(
                            sample["turn_id"]
                        )

        # Embedding size
        self.embed_size = len(
            self.samples_with_item_targets["test"]["embedding"][0]
        )

        # Single token to llm tokens
        mapping = json.load(
            open(
                os.path.join(
                    args.data_dir, "item_entity_name_to_item_indices.json"
                ),
                "r",
            )
        )
        self.single2llm = {
            mapping[name]["single_token"]: mapping[name]["llm_tokens"]
            for name in mapping.keys()
        }

        # Convert data2single to list
        self.data2single = {}
        for data_name in args.data_names:
            path = os.path.join(args.data_dir, data_name, "id2name.json")
            names = json.load(open(path, "r")).values()
            self.data2single[data_name] = [
                mapping[name]["single_token"] for name in names
            ]

        # Number of items
        self.num_items = (
            max(
                max(self.data2single[data_name])
                for data_name in args.data_names
            )
            + 1
        )

        # Build the single2llm as a tensor with pad = 0
        self.single2llm = torch.nn.utils.rnn.pad_sequence(
            [
                torch.LongTensor(self.single2llm[i])
                for i in range(self.num_items)
            ],
            batch_first=True,
            padding_value=0,
        )


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, mode, args):
        self.samples_with_item_targets = data.samples_with_item_targets[mode]
        self.args = args
        self.mode = mode

    def __len__(self):
        return len(self.samples_with_item_targets["embedding"])

    def __getitem__(self, index):
        batch = {
            "input": torch.FloatTensor(
                self.samples_with_item_targets["embedding"][index]
            ),
            "label": self.samples_with_item_targets["single_token"][index],
            "dataset": self.samples_with_item_targets["dataset"][index],
        }
        return batch


## Define Model


class Model(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        # save hyperparameters
        self.save_hyperparameters()
        # args
        self.args = args
        # model
        self.model = AGGREGATORS[args.aggregator](args)
        self.single2llm = nn.Embedding.from_pretrained(
            args.single2llm.float(), freeze=True
        )
        self.llm_embed = nn.Embedding.from_pretrained(
            torch.load(args.llm_embed_path), freeze=args.freeze_llm_embed
        )
        # loss defintion
        self.loss = nn.CrossEntropyLoss()
        # evaluation results
        self.step_outputs = defaultdict(list)
        self.step_outputs_by_dataset = {
            data: defaultdict(list) for data in self.args.data_names
        }

    def squeeze_embeddings(self, candidates):
        """Squeeze the llm token embedding to single token embedding

        Args:
            candidates: candidate indices with size (candidate_size, )

        Returns:
            embeddings: embeddings with size (candidate_size, embed_size)
        """
        llm_tokens = (
            self.single2llm(candidates).detach().long()
        )  # (candidate_size, llm_len)
        llm_embed = self.llm_embed(
            llm_tokens
        )  # (candidate_size, llm_len, embed_size)
        attention_mask = llm_tokens != 0  # (candidate_size, llm_len)
        return self.model(
            inputs_embeds=llm_embed, attention_mask=attention_mask
        )  # (candidate_size, embed_size)

    def update_single_embed(self):
        """Update the single token embedding using the llm token embedding"""
        self.single_embed = []

        # chunk size is args.embed_update_batch_size
        for i in tqdm(
            range(0, self.args.num_items, self.args.embed_update_batch_size),
            desc="Update Item Embeddings",
        ):
            candidates = torch.arange(
                i,
                min(i + self.args.embed_update_batch_size, self.args.num_items),
            ).to(self.device)  # (batch_size)
            embeddings = self.squeeze_embeddings(
                candidates
            )  # (batch_size, embed_size)
            self.single_embed.append(embeddings)

        # concatenate
        self.single_embed = torch.cat(
            self.single_embed, dim=0
        )  # (num_items, embed_size)

    def forward(self, q, pos=None, neg=None):
        """
        Args:
            q: query embedding with size (batch_size, embed_size)
            pos: positive item indices with size (batch_size, )
            neg: negative item indices with size (negs, )

        Returns:
            logits: logits with size (batch_size, 1 + self.args.negs) when candidates is not None
                or logits with size (batch_size, self.args.num_items) when candidates is None
        """
        # Embedding
        if pos is None or neg is None:
            embed = self.single_embed  # (num_items, embed_size)
            logits = q @ embed.T  # (batch_size, num_items)
        else:
            # Get the positive logits
            embed = self.squeeze_embeddings(pos)  # (batch_size, embed_size)
            pos_logits = (q * embed).sum(dim=-1)  # (batch_size, )

            # Get the negative logits
            embed = self.squeeze_embeddings(neg)  # (args.negs, embed_size)
            neg_logits = q @ embed.T  # (batch_size, args.negs)

            # Concatenate
            logits = torch.cat(
                [pos_logits.unsqueeze(1), neg_logits], dim=1
            )  # (batch_size, 1 + args.negs)
        return logits

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.args.lr, weight_decay=self.args.decay
        )

    def eval_step(self, batch, mode="test"):
        # Get the logits
        q, l, d = (
            batch["input"].float(),
            batch["label"].long(),
            batch["dataset"],
        )
        logits = self.forward(q)
        pos_logits = logits[torch.arange(len(q)), l].unsqueeze(
            1
        )  # (batch_size, 1)

        # Update evaluation results
        rank = torch.div(
            (logits > pos_logits).sum(dim=1)
            + (logits >= pos_logits).sum(dim=1),
            2,
            rounding_mode="floor",
        )
        self.step_outputs[mode].extend(rank.tolist())

        # Update evaluation results by dataset
        for i, data in enumerate(d):
            logits_ = logits[i, self.args.data2single[data]]
            pos_logits_ = pos_logits[i, 0]
            rank_ = torch.div(
                (logits_ > pos_logits_).sum() + (logits_ >= pos_logits_).sum(),
                2,
                rounding_mode="floor",
            )
            self.step_outputs_by_dataset[data][mode].append(rank_.item())

    def _on_epoch_end(self, mode):
        # Compute metrics
        metrics = ranks2metrics(self.step_outputs[mode], self.args.ks)
        # Log metrics
        for k, v in metrics.items():
            self.log(f"{mode}/{k}", v)
        # Reset evaluation results
        self.step_outputs[mode].clear()

        # Compute metrics by dataset
        for data in self.args.data_names:
            metrics = ranks2metrics(
                self.step_outputs_by_dataset[data][mode], self.args.ks
            )
            # Log metrics
            for k, v in metrics.items():
                self.log(f"{mode}/{data}/{k}", v)
            # Reset evaluation results
            self.step_outputs_by_dataset[data][mode].clear()

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, "valid")

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, "test")

    def on_validation_epoch_start(self):
        self.update_single_embed()

    def on_test_epoch_start(self):
        self.update_single_embed()

    def on_validation_epoch_end(self):
        self._on_epoch_end("valid")

    def on_test_epoch_end(self):
        self._on_epoch_end("test")


## Define Training


def test(args, label):
    # Load data and build dataloader-friendly data.samples
    args.label = label
    data = Data(args=args)

    # Update args
    args.num_items = data.num_items
    args.embed_size = data.embed_size
    args.single2llm = data.single2llm
    args.data2single = data.data2single

    # Build dataloader
    val_loader = DataLoader(
        Dataset(data=data, mode="valid", args=args),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=16,
    )
    test_loader = DataLoader(
        Dataset(data=data, mode="test", args=args),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=16,
    )

    # model and trainer
    model = Model(args)
    print(model)
    early_stop_callback = EarlyStopping(
        monitor="valid/Recall@10", patience=100, verbose=False, mode="max"
    )
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1, monitor="valid/Recall@10", mode="max", filename="best"
    )
    trainer = pl.Trainer(
        default_root_dir=args.ckpt_dir,
        accelerator="auto",
        max_epochs=args.epochs,
        max_time={"minutes": args.max_minutes},
        gradient_clip_val=5.0,
        callbacks=[early_stop_callback, checkpoint_callback],
        check_val_every_n_epoch=args.print_every,
        log_every_n_steps=min(len(test_loader), 10000),
        logger=CSVLogger(
            args.ckpt_dir,
            name=os.path.join(
                "_".join(args.data_names).replace("/", "_"),
                args.aggregator,
                args.label,
            ),
        ),
    )

    # mode fit or test
    model.load_state_dict(
        torch.load(args.load_pretrained_weights)["state_dict"]
    )
    trainer.validate(model=model, dataloaders=val_loader)
    trainer.test(model=model, dataloaders=test_loader)


def main(
    data_dir: str = "data",
    llm_embed_path: str = "data/llm_embedding.pt",
    freeze_llm_embed: bool = True,
    aggregator: str = "rnn",
    dropout_prob: float = 0.1,
    data_names: list = ["inspired", "redial", "redditv1.5"],
    epochs: int = 200,
    batch_size: int = 256,
    embed_update_batch_size: int = 2048,
    lr: float = 0.001,
    decay: float = 0,
    negs: int = 1000,
    print_every: int = 2,
    ckpt_dir: str = "logs/test",
    load_pretrained_weights: str = None,
    test_only: bool = False,
    max_minutes: int = 180,
    ks: list = [1, 5, 10, 50],
):
    # Load args
    args = Namespace(**locals())
    print(args)

    # Test with label from llm as single token
    test(args, label="labels_from_llm_as_single_token")

    # Test with label from data as single token
    test(args, label="labels_from_data_as_single_token")


if __name__ == "__main__":
    CLI(main)
