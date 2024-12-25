# SCE
The code and data of the SCE framework.

# Enviroment
We recommend using Anaconda to set up a virtual environment with `python=3.10` and `pytorch-cuda=12.1`, and installing the `transformers` and `vllm` packages. If prompted to install additional packages during setup, use `pip` to install the default versions.

# Quik Start
Using the evaluation of Qwen2-7B on SemEval as an example.

1) Download the Qwen2-7B-Instruct model from Hugging Face (https://huggingface.co/Qwen/Qwen2-7B-Instruct) and place it in the `../llms` folder.
2) Run `store_cause_concepts.py` to identify the cause concepts for label concepts, i.e., discovering confounders.
3) 
