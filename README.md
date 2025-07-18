# SCE
The code and data of the SCE (Spurious Correlation Evaluator) framework, the first work to evaluate the causal stability of LLMs. Using the evaluation of Qwen2-7B on SemEval as an example, this repository includes detailed reproduction instructions and generated intermediate files within the corresponding folders.

# Datasets
The `datasets` folder contains three datasets used in our experiments: SemEval for relation extraction, Few-NERD for entity typing, and ACE 2005 for event detection.

# Enviroment
We recommend using Anaconda to set up a virtual environment with `python=3.10` and `pytorch-cuda=12.1`, and installing the `transformers` and `vllm` packages. If prompted to install additional packages during setup, use `pip` to install the default versions.

# Quik Start
Using the evaluation of Qwen2-7B on SemEval as an example.

1) Download the Qwen2-7B model from Hugging Face (https://huggingface.co/Qwen/Qwen2-7B-Instruct-GPTQ-Int8) and place it in the `../llms` folder.
2) Run `store_cause_concepts.py` to identify the cause concepts for label concepts, i.e., discovering confounders. The results are saved in the `cause_concepts` folder.
3) Run `main.py` to evaluate the causal stability of Qwen2-7B on SemEval. The results are saved in the `evaluation_results` folder.
4) Run `result_analysis.py` to output the final results.

