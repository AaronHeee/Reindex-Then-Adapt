from __future__ import annotations
import sys, json, time, os


# Get the absolute path of the current directory
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(CUR_DIR, "..")

# Append the path of the tools directory to the system path
sys.path.append(os.path.join(ROOT_DIR, "tools/exllamav2/examples"))

# From third-party libraries
from jsonargparse import CLI


def main(
    data_dir: str = "data",
    data_name: str = "inspired/test.jsonl",
    model_dir: str = "ckpts/Llama2-7B-chat-exl2",
    prompt_format: str = "llama",
    system_prompt: str = "Pretend you are a movie recommender system. \nI will give you a conversation between a user and you (a recommender system). Based on the conversation, you reply me with 20 movie titles without extra sentences.\n",
    prompt_template: str = "Here is the conversation: {}",
    max_response_len: int = 512,
    max_batch_size: int = 1024,
    max_q_size: int = 1,
    cache_size: int = 32768,
    from_row: int = 0,
    to_row: int = 1000,
    out_file: str = "reindex_step/data/inspired/test/output.jsonl",
    temperature: float = 0,
    top_k: int = 0,
    top_p: float = 0,
    token_repetition_penalty: float = 1.0,
):
    from util import format_prompt, get_stop_conditions
    from exllamav2 import (
        ExLlamaV2,
        ExLlamaV2Config,
        ExLlamaV2Cache,
        ExLlamaV2Tokenizer,
    )
    from exllamav2.generator import (
        ExLlamaV2DynamicGenerator,
        ExLlamaV2DynamicJob,
        ExLlamaV2Sampler,
    )

    # Out file directory
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    # Create model and generator

    config = ExLlamaV2Config(model_dir)
    model = ExLlamaV2(config)
    cache = ExLlamaV2Cache(model, max_seq_len=cache_size, lazy=True)
    model.load_autosplit(cache, progress=True)
    tokenizer = ExLlamaV2Tokenizer(config)

    generator = ExLlamaV2DynamicGenerator(
        model=model,
        cache=cache,
        tokenizer=tokenizer,
        max_batch_size=max_batch_size,
        max_q_size=max_q_size,
    )

    gen_settings = ExLlamaV2Sampler.Settings(
        token_repetition_penalty=token_repetition_penalty,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    # Load a dataset of prompts, print the first couple of entries

    dataset_list = [
        eval(l)
        for l in open(
            os.path.join(data_dir, data_name),
            "r",
            encoding="utf-8",
            errors="ignore",
        )
    ]
    if to_row > 0 and from_row < to_row:
        dataset_list = dataset_list[from_row:to_row]

    print()
    print(f"Dataset loaded, {len(dataset_list)} rows:")
    print()
    for i in range(min(5, len(dataset_list))):
        print(f"{i}: {dataset_list[i]}")

    # Create job list

    print()
    print("Creating jobs...")

    completions = []

    input_records = []
    for idx, p in enumerate(dataset_list):
        input_records.append(p)
        prompt = prompt_template.format(
            p["context"]
            .encode("utf-8", errors="ignore")
            .decode("utf-8", errors="ignore")[:2000]
        )
        f_prompt = format_prompt(prompt_format, system_prompt, prompt)
        completions.append(f_prompt)
        prompt_ids = tokenizer.encode(f_prompt, encode_special_tokens=True)

        job = ExLlamaV2DynamicJob(
            input_ids=prompt_ids,
            gen_settings=gen_settings,
            max_new_tokens=max_response_len,
            identifier=idx,
            stop_conditions=get_stop_conditions(prompt_format, tokenizer),
        )
        generator.enqueue(job)
        if (idx + 1) % 1000 == 0 or (idx + 1) == len(dataset_list):
            print(f"{idx + 1} / {len(dataset_list)}")

    # Generate

    print()
    print("Generating...")

    num_completions = 0
    num_tokens = 0
    time_begin = time.time()

    while generator.num_remaining_jobs():
        results = generator.iterate()

        # We'll always get at least one result for each active job, even if the result contains no output text
        bsz = len(set([r["identifier"] for r in results]))

        for result in results:
            if not result["eos"]:
                continue

            # EOS signal is always accompanied by the full completion, so we don't need to collect text chunks
            idx = result["identifier"]
            response = result["full_completion"]
            completions[idx] += response

            # Measure performance
            num_completions += 1
            num_tokens += result["new_tokens"]
            elapsed_time = time.time() - time_begin
            rpm = num_completions / (elapsed_time / 60)
            tps = num_tokens / elapsed_time
            print()
            print(
                "---------------------------------------------------------------------------"
            )
            print(f"Current batch size: {bsz}")
            print(f"Avg. completions/minute: {rpm:.2f}")
            print(f"Avg. output tokens/second: {tps:.2f}")
            print(f"Num. completions: {num_completions} / {len(dataset_list)}")
            print(
                "---------------------------------------------------------------------------"
            )

    # Save output

    with open(out_file, "w", encoding="utf-8") as f:
        for idx, completion in enumerate(completions):
            entry = input_records[idx].copy()
            entry["completion"] = completion
            f.write(
                json.dumps(entry, ensure_ascii=False)
                .encode("utf-8", errors="ignore")
                .decode("utf-8", errors="ignore")
                + "\n"
            )


if __name__ == "__main__":
    CLI(main)
