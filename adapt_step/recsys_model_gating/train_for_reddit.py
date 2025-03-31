# Define Dependency

import os
import sys
import json
import mmap
import random

from collections import defaultdict

# Get the absolute path of the current directory
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(CUR_DIR, "../..")

# Add the root directory to the system path
sys.path.append(ROOT_DIR)
from tools.evaluator import ranks2metrics
from reindex_step.aggregators import AGGREGATORS
from adapt_step.recsys import RECSYS

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


class FastJSONLSampler:
    def __init__(
        self,
        jsonl_file_path,
        data_name,
        label="labels_from_llm_as_single_token",
    ):
        self.file_path = jsonl_file_path

        # Create memory map and find line positions
        with open(jsonl_file_path, "rb") as f:
            self.mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        # Store line offsets for fast random access
        self.line_offsets = [0]
        offset = 0
        while True:
            offset = self.mm.find(b"\n", offset) + 1
            if offset == 0:
                break
            self.line_offsets.append(offset)

        self.line_count = len(self.line_offsets) - 1
        self.data_name = data_name
        self.label = label

    def get_single_line(self, line_number):
        start = self.line_offsets[line_number]
        end = self.line_offsets[line_number + 1] - 1
        line = json.loads(self.mm[start:end].decode("utf-8"))

        sampled_label = random.choice(line[self.label])

        return {
            "embedding": line["encoding"],
            "single_token": sampled_label,
            "dataset": self.data_name,
            "conv_id": str(line["conv_id"]),
            "turn_id": str(line["turn_id"]),
        }

    def __del__(self):
        self.mm.close()

    def close(self):
        self.mm.close()


## Define Dataset


class Data(object):
    def __init__(self, args):
        """Build data according to args.data_names

        Attributes:
            samples_with_item_targets: train / valid / test samples with item targets
                - Example: x['train']['embedding'],  x['train']['single_token'], x['train']['conv_id'], x['train']['turn_id']

            embed_size: int, embedding size
            num_items: int, number of items

            single2llm: dict, mapping from single token to llm tokens
            data2single: dict, mapping from data to the list of single token within the data
        """
        self.args = args

        print("Loading data for recsys...")
        self.load_data_for_recsys(args)

        print(
            "Building the `samples_with_item_targets` used for training, validation, and testing..."
        )

        self.samples_with_item_targets = {}

        # Load validation and testing samples
        for data_name in args.data_names:
            for split in ["valid", "test"]:
                path = os.path.join(
                    ROOT_DIR, "data", data_name, f"{split}-for-adapt.jsonl"
                )
                for line in tqdm(
                    open(path, "r", encoding="utf-8"),
                    desc=f"loading {data_name} {split}:",
                ):
                    sample = json.loads(line)
                    for sing_token in sample[self.args.label]:
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
                            str(sample["conv_id"])
                        )
                        self.samples_with_item_targets[split]["turn_id"].append(
                            str(sample["turn_id"])
                        )

        # Load training samples
        assert len(args.data_names) == 1, (
            "Only support one data_name for training now"
        )

        data_name = args.data_names[0]
        print(f"Loading training samples from {data_name}...")

        path = os.path.join(
            ROOT_DIR, "data", data_name, "train-for-adapt.jsonl"
        )
        self.train_sampler_with_item_targets = FastJSONLSampler(
            path, data_name=data_name, label=args.label
        )

        # Embedding size
        self.embed_size = self.args.embed_size

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

    def load_data_for_recsys(self, args):
        # Build (conv_id, turn_id) to historical items and item mapping per dataset

        mapping = defaultdict(dict)
        for data_name in args.data_names:
            for l in open(
                os.path.join(
                    args.data_dir,
                    data_name,
                    "conv_id_turn_id_to_item_history.jsonl",
                ),
                "r",
                encoding="utf-8",
            ):
                l = eval(l)
                mapping[data_name][l["conv_id"], l["turn_id"]] = l[
                    "history_item"
                ]

        self.history = mapping


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, mode, args):
        self.samples_with_item_targets = data.samples_with_item_targets[mode]
        self.history = data.history
        self.args = args
        self.mode = mode

    def __len__(self):
        return len(self.samples_with_item_targets["embedding"])

    def __getitem__(self, index):
        data_name, conv_id, turn_id = (
            self.samples_with_item_targets["dataset"][index],
            self.samples_with_item_targets["conv_id"][index],
            self.samples_with_item_targets["turn_id"][index],
        )
        history = self.history[data_name][conv_id, turn_id]
        batch = {
            "input": torch.FloatTensor(
                self.samples_with_item_targets["embedding"][index]
            ),
            "recsys_input": torch.LongTensor(
                [self.args.cls_token]
                + history[-self.args.max_len :]
                + [self.args.pad_token] * (self.args.max_len - len(history))
            ),
            "label": self.samples_with_item_targets["single_token"][index],
            "dataset": self.samples_with_item_targets["dataset"][index],
        }
        return batch


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, data, mode, args):
        self.sampler_with_item_targets = data.train_sampler_with_item_targets
        self.history = data.history
        self.args = args
        self.mode = mode

    def __len__(self):
        return self.sampler_with_item_targets.line_count

    def __getitem__(self, index):
        line = self.sampler_with_item_targets.get_single_line(index)
        data_name, conv_id, turn_id = (
            line["dataset"],
            line["conv_id"],
            line["turn_id"],
        )
        history = self.history[data_name][conv_id, turn_id]
        batch = {
            "input": torch.FloatTensor(line["embedding"]),
            "recsys_input": torch.LongTensor(
                [self.args.cls_token]
                + history[-self.args.max_len :]
                + [self.args.pad_token] * (self.args.max_len - len(history))
            ),
            "label": line["single_token"],
            "dataset": line["dataset"],
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

        # pre-trained aggregator model
        if args.aggregator_path is not None:
            self.model = self._load_pretrained_aggregator(args)
        else:
            self.model = AGGREGATORS[args.aggregator](args)

        # recsys model
        self.recsys = RECSYS[args.recsys](args)

        self.single2llm = nn.Embedding.from_pretrained(
            args.single2llm.float(), freeze=True
        )
        self.llm_embed = nn.Embedding.from_pretrained(
            torch.load(args.llm_embed_path), freeze=args.freeze_llm_embed
        )

        # coeff
        self.coeff = nn.Parameter(torch.tensor(0.0))

        # loss defintion
        self.loss = nn.CrossEntropyLoss()
        # evaluation results
        self.step_outputs = defaultdict(list)
        self.step_outputs_by_dataset = {
            data: defaultdict(list) for data in self.args.data_names
        }

    def _load_pretrained_aggregator(self, args):
        """Load the pre-trained aggregator model"""

        # Load model state dict
        model_path = args.aggregator_path
        model_state_dict = torch.load(model_path)

        # Initialize the model
        model_args = model_state_dict["hyper_parameters"]["args"]
        self.model = AGGREGATORS[model_args.aggregator](model_args)

        # Load the model state dict
        self.load_state_dict(model_state_dict["state_dict"], strict=False)

        # Freeze the model
        if self.args.freeze_aggregator:
            for param in self.model.parameters():
                param.requires_grad = False

        return self.model

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

    def forward(self, h, q, pos=None, neg=None):
        """
        Args:
            h: historical items with size (batch_size, max_len)
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
            q_logits = q @ embed.T  # (batch_size, num_items)

            # Add recsys
            h_logits = self.recsys(h)  # (batch_size, num_items)

            # Gating
            coeff = self.coeff.sigmoid()
            logits = coeff * q_logits + (1 - coeff) * h_logits

        else:
            # Get the positive logits
            embed = self.squeeze_embeddings(pos)  # (batch_size, embed_size)
            pos_logits = (q * embed).sum(dim=-1).unsqueeze(1)  # (batch_size, 1)

            # Get the negative logits
            embed = self.squeeze_embeddings(neg)  # (args.negs, embed_size)
            neg_logits = q @ embed.T  # (batch_size, args.negs)

            # Recsys results
            h_logits = self.recsys(h)  # (batch_size, num_items)

            # Gating
            pos_h_logits = h_logits[torch.arange(len(q)), pos].unsqueeze(
                1
            )  # (batch_size, 1)
            neg_h_logits = h_logits[:, neg]  # (batch_size, args.negs)

            coeff = self.coeff.sigmoid()

            pos_logits = coeff * pos_logits + (1 - coeff) * pos_h_logits
            neg_logits = coeff * neg_logits + (1 - coeff) * neg_h_logits

            # Concatenate
            logits = torch.cat(
                [pos_logits, neg_logits], dim=1
            )  # (batch_size, 1 + args.negs)

        return logits

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.args.lr, weight_decay=self.args.decay
        )

    def eval_step(self, batch, mode="train"):
        # Get the logits
        h, q, l, d = (
            batch["recsys_input"].long(),
            batch["input"].float(),
            batch["label"].long(),
            batch["dataset"],
        )
        logits = self.forward(h, q)
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

    def training_step(self, batch, batch_idx):
        h, q, l = (
            batch["recsys_input"].long(),
            batch["input"].float(),
            batch["label"].long(),
        )
        n = torch.randint(0, self.args.num_items, (self.args.negs,)).to(
            l.device
        )
        logits = self.forward(h, q, pos=l, neg=n)
        loss = self.loss(logits, torch.zeros_like(l).to(l.device))
        self.log("train/loss", loss.item(), prog_bar=True)
        return loss

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

        self.log(f"{mode}/coeff", self.coeff.sigmoid().item())

    def on_validation_epoch_start(self):
        self.update_single_embed()

    def on_test_epoch_start(self):
        self.update_single_embed()

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, "valid")

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, "test")

    def on_validation_epoch_end(self):
        self._on_epoch_end("valid")

    def on_test_epoch_end(self):
        self._on_epoch_end("test")


## Define Training


def main(
    data_dir: str = "data",
    llm_embed_path: str = "data/llm_embedding.pt",
    label: str = "labels_from_data_as_single_token",
    freeze_llm_embed: bool = True,
    freeze_aggregator: bool = True,
    aggregator: str = "rnn",
    recsys: str = "fism",
    recsys_with_bias: bool = True,
    recsys_embed_size: int = 64,
    alpha: float = 0.5,
    max_len: int = 128,
    dropout_prob: float = 0.1,
    data_names: list = ["redditv1.5"],
    embed_size: int = 4096,
    epochs: int = 200,
    batch_size: int = 256,
    embed_update_batch_size: int = 2048,
    lr: float = 0.001,
    decay: float = 0,
    negs: int = 1000,
    print_every: int = 2,
    ckpt_dir: str = "logs",
    aggregator_path: str = "ckpts/best_aggregator/checkpoints/best.ckpt",
    load_pretrained_weights: str = None,
    test_only: bool = False,
    max_minutes: int = 180,
    ks: list = [1, 5, 10, 50],
):
    # Load args
    args = Namespace(**locals())
    print(args)

    # Load data and build dataloader-friendly data.samples
    data = Data(args=args)

    # Update args
    args.num_items = data.num_items
    args.cls_token = data.num_items
    args.pad_token = data.num_items + 1

    args.embed_size = data.embed_size
    args.single2llm = data.single2llm
    args.data2single = data.data2single

    # Build dataloader
    train_loader = DataLoader(
        TrainDataset(data=data, mode="train", args=args),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=16,
    )
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
        monitor="valid/Recall@10", patience=3, verbose=False, mode="max"
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
        log_every_n_steps=min(len(train_loader), 10000),
        logger=CSVLogger(
            args.ckpt_dir,
            name=os.path.join(
                "_".join(args.data_names).replace("/", "_"),
                args.recsys,
            ),
        ),
    )

    # mode fit or test
    if args.test_only:
        model.load_state_dict(
            torch.load(args.load_pretrained_weights)["state_dict"]
        )
        trainer.test(model=model, dataloaders=test_loader)
    else:
        trainer.test(model=model, dataloaders=test_loader)
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
        trainer.test(model=model, dataloaders=test_loader, ckpt_path="best")

    data.train_sampler_with_item_targets.close()


if __name__ == "__main__":
    CLI(main)
