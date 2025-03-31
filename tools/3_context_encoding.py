import os
import json

# From third-party libraries
from jsonargparse import CLI
import torch
import transformers
from tqdm import tqdm


def process_context(context, START_WORD="\n1.", INST_WORD="[/INST] "):
    if context.find(START_WORD) != -1:
        return context[: context.find(START_WORD) + len(START_WORD)]
    else:
        return context[: context.find(INST_WORD) + len(INST_WORD)]


def main(
    data_dir: str = "reindex_step/data/inspired/test",
    data_name: str = "output-mapping.jsonl",
    device: str = "cuda:0",
    model_name: str = "meta-llama/Llama-2-7b-chat-hf",
    field: str = "completion",
    batch_size: int = 4,
    max_context_len: int = 512,
    n: int = 0,
    N: int = 1,
):
    # set save dir
    save_dir = data_dir
    # load model
    config = transformers.AutoConfig.from_pretrained(
        model_name, trust_remote_code=True
    )
    model = (
        transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.bfloat16,  # Load model weights in bfloat16
            trust_remote_code=True,
        )
        .to(device)
        .eval()
    )

    with torch.no_grad():
        # load tokenizer
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = tokenizer.unk_token_id
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"

        # read jsonl file
        data_list = [
            json.loads(l.strip())
            for l in open(os.path.join(data_dir, data_name), "r")
        ]

        # load the n-th part of the data if N > 1
        if N > 1:
            length = len(data_list)
            start = length * n // N
            end = length * (n + 1) // N
            data_list = data_list[start:end]

        # encode samples
        with open(
            os.path.join(
                save_dir, f"{data_name.split('.')[0]}-encoding-{n}-{N}.jsonl"
            ),
            "w",
            encoding="utf-8",
        ) as f:
            length = len(data_list)
            for i in tqdm(
                range(0, length, batch_size),
                desc=f"Encoding samples from {data_name}",
            ):
                # batchify
                batch = data_list[i : i + batch_size]
                contexts = [process_context(sample[field]) for sample in batch]
                inputs = tokenizer(
                    contexts,
                    truncation=True,
                    max_length=max_context_len,
                    padding="longest",
                    return_tensors="pt",
                ).to(model.device)

                inputs = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "position_ids": (
                        inputs["attention_mask"].cumsum(dim=-1) - 1
                    ).relu(),
                }

                # inference
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    hidden_states = model.model(**inputs).last_hidden_state

                # save
                encodings = hidden_states[:, -1, :].tolist()

                for sample, encoding, context in zip(
                    batch, encodings, contexts
                ):
                    sample.update(
                        {"encoding": encoding, "encoded_text": context}
                    )
                    f.write(json.dumps(sample) + "\n")


if __name__ == "__main__":
    CLI(main)
