# Define Dependency

import os
import sys

# Get the absolute path of the current directory
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = CUR_DIR

# Add the root directory to the system path
sys.path.append(ROOT_DIR)

# From third-party libraries
import torch
from torch import nn


class FISM(nn.Module):
    def __init__(self, args):
        super().__init__()

        # args
        self.args = args

        # model
        self.model = nn.Embedding(
            args.num_items + 2,
            args.recsys_embed_size,
            padding_idx=self.args.pad_token,
        )
        self.model.weight.data.normal_(0, 0.01)

        # bias
        self.bias = nn.Parameter(torch.zeros(args.num_items + 2))
        self.bias.data.zero_()

    def forward(self, h):
        """
        Args:
            h: historical items with size (batch_size, max_len)

        Returns:
            logits: logits with size (batch_size, num_items)
        """
        # Embedding
        embed = self.model(h)  # (batch_size, max_len, recsys_embed_size)
        # Average (with alpha) pooling but ignore the padding tokens
        embed = (
            embed.sum(dim=1)
            / (h != self.args.pad_token).sum(dim=1).unsqueeze(1).float()
            ** self.args.alpha
        )  # (batch_size, recsys_embed_size)
        # Logits
        logits = embed @ self.model.weight.T  # (batch_size, num_items + 2)
        if self.args.recsys_with_bias:
            logits += self.bias.unsqueeze(dim=0)  # (batch_size, num_items + 2)
        return logits[:, :-2]  # (batch_size, num_items)


class SASRec(nn.Module):
    def __init__(self, args):
        super().__init__()

        # use the GPT2 model from Huggingface as a SASRec-like implementation
        from transformers import GPT2Config, GPT2Model

        # args
        self.args = args

        # model
        config = GPT2Config(
            vocab_size=args.num_items + 2,
            n_positions=args.max_len + 1,
            n_ctx=args.max_len,
            n_embd=args.recsys_embed_size,
            n_layer=2,
            n_head=2,
            activation_function="gelu",
            resid_pdrop=args.dropout_prob,
            embd_pdrop=args.dropout_prob,
            attn_pdrop=args.dropout_prob,
        )
        self.model = GPT2Model(config)
        self.embed = self.model.wte

        # bias
        self.bias = nn.Parameter(torch.zeros(args.num_items + 2))
        self.bias.data.zero_()

    def forward(self, h):
        """
        Args:
            h: historical items with size (batch_size, max_len)

        Returns:
            logits: logits with size (batch_size, num_items)
        """
        # Mask for padding tokens
        mask = h != self.args.pad_token

        # Format input into huggingface style
        inputs = {"input_ids": h, "attention_mask": mask}

        # Get contextual token embeddings
        h = self.model(**inputs)[0]  # (batch_size, max_position, hidden_size)

        # Pick up the last token embeddings
        h = h[torch.arange(h.size(0)), mask.sum(dim=1) - 1]

        # Logits
        logits = h @ self.embed.weight.T
        if self.args.recsys_with_bias:
            logits += self.bias.unsqueeze(dim=0)  # (batch_size, num_items + 2)

        return logits[:, :-2]  # (batch_size, num_items)


RECSYS = {
    "fism": FISM,
    "sasrec": SASRec,
}
