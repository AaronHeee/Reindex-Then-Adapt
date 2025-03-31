# Define Dependency
import torch
from torch import nn


class SumAggregator(nn.Module):
    def __init__(self, args):
        super().__init__()

    def forward(self, inputs_embeds, attention_mask):
        """Sum up the embeddings"""
        return torch.sum(inputs_embeds * attention_mask.unsqueeze(-1), dim=1)


class WeightedAggregator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(1, args.single2llm.size(1), 1))
        self.proj = nn.Linear(args.embed_size, args.embed_size)

    def forward(self, inputs_embeds, attention_mask):
        """Weighted sum up the embeddings with softmax weights"""
        weights = self.weights.repeat(inputs_embeds.size(0), 1, 1)
        weights.masked_fill_(attention_mask.unsqueeze(-1) == 0, float("-inf"))
        weights = weights.softmax(dim=1)
        x = torch.sum(inputs_embeds * weights, dim=1)
        return self.proj(x)


class RNNAggregator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hidden_size = args.embed_size // 4
        self.rnn = nn.GRU(
            input_size=args.embed_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Linear(self.hidden_size * 2, args.embed_size)

    def forward(self, inputs_embeds, attention_mask):
        """Pick up the last hidden state of the RNN given the attention mask"""
        x, _ = self.rnn(inputs_embeds)
        x = x[torch.arange(x.size(0)), attention_mask.sum(dim=1) - 1]
        return self.proj(x)


class TRMAggregator(nn.Module):
    def __init__(self, args):
        super().__init__()

        # build a transformer decoder layer with pytorch
        self.trm = nn.TransformerEncoderLayer(
            d_model=args.embed_size,
            nhead=8,
            dropout=args.dropout_prob,
            batch_first=True,
        )

    def forward(self, inputs_embeds, attention_mask):
        """Pick up the last hidden state of the TRM given the attention mask"""
        x = self.trm(inputs_embeds)
        return x[torch.arange(x.size(0)), attention_mask.sum(dim=1) - 1]


AGGREGATORS = {
    "sum": SumAggregator,
    "weighted": WeightedAggregator,
    "rnn": RNNAggregator,
    "trm": TRMAggregator,
}
