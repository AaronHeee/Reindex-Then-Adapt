# ExllamaV2 Setup

This [tool](https://github.com/turboderp-org/exllamav2) is used to do LLM (llama2) inference efficiently. If you want to use our generated data for the Reindex step, please download the data in our [README.md](../../README.md) directly.

## Dependency

It is a bit tricky to set up the ExllamaV2 on RTA 3090. Sharing our steps for reference:

1. Create a new conda environment:

```shell
conda create -n exllamav2_env python=3.11
conda activate exllamav2_env
```

2. Install ExllamaV2, this is the version working for our RTX 3090:

```shell
pip install https://github.com/turboderp-org/exllamav2/releases/download/v0.2.8/exllamav2-0.2.8+cu124.torch2.6.0-cp311-cp311-linux_x86_64.whl

pip install rich tokenizers Pillow datasets 
```

3. Install the compatible CUDA compiler and Flash Attention:

```shell
conda install cuda-compiler=12.4 -c nvidia
pip install flash-attn
```

4. Download the Llama2-7B model and test it:

```shell
ckpt=/path/to/ckpts/Llama2-7B-chat-exl2

CUDA_VISIBLE_DEVICES=0 python test_inference.py -m ${ckpt} -p "Once upon a time,"
```

```
-- Model: /path/tockpts/Llama2-7B-chat-exl2
 -- Options: []
 -- Loading model...
 -- Loaded model in 102.8959 seconds
 -- Loading tokenizer...
 -- Warmup...
 -- Generating...

Once upon a time, there was a man named Nnamdi who lived in a small village in Nigeria. He was a skilled craftsman and spent most of his days working on his pottery. He was particularly known for his beautiful vases and jars, which he crafted with great care and attention to detail.
One day, a wealthy merchant came to the village in search of talented craftsmen to create a special gift for his queen. The merchant had heard of Nnamdi's exceptional skills and asked him to create a beautiful jar for the queen.
Nnamdi was honored and agreed to

 -- Response generated in 1.52 seconds, 128 tokens, 84.43 tokens/second (includes prompt eval.)
 ```

