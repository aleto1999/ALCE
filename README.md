# RAGGED and ALCE


# uh what
## Quick Links

  - [Requirements](#requirements)
  - [Code Structure](#code-structure)
  - [Data and Setup](#data-and-setup)

  - [Reproducing Baselines](#reproducing-baselines)
  - [Evaluation](#evaluation)
  - [Human Evaluation](#human-evaluation)
  - [Bug or Questions](#bug-or-questions)
  - [Citation](#citation)


## Requirements

Please install the latest versions of PyTorch (`torch`), HuggingFace Transformers (`transformers`), HuggingFace Accelerate (`accelerate`), and the OpenAI API package (`openai`). This codebase is tested on 
`torch==2.1.0.dev20230514+cu118`, `transformers==4.28.1`, `accelerate==0.17.1`, and `openai==0.27.4` with Python 3.9.7.
# TODO: add to this list


## Code Structure
* `setup`: directory containing scripts for downloading data and setting env variables
* `retrieve`: directory containing scripts and files for retrieval and retriever eval
  * `configs`: config files for different retrieval settings
  * `index.py`: index datasets before or during retrieval step
  * `run.py`: retrieval, pass config file
  * `eval.py`:   
* `read`: directory containing scripts and files for generation and generation eval
  * `configs`: config files for different generation settings
  * `prompts`: folder that contains all prompt files.
  * `run.py`: retrieval, pass config file
  * `eval.py`: eval file to evaluate generations.
* `tools/`: misc code (generate summaries/snippets, reranking, etc.)

## Data and Setup

You can download datasets (along with retrieval results) by running the following command:

```bash
bash setup/download_data.sh
```

All the data will be stored in `data/`.

Before running any scripts, set up necessary environment variables with:

```bash
bash setup/set_paths.sh
```
This sets up routing for all retrieval and runner output. By default, everything will be output to `data/`, but this can be easily changed. 


### Retrieval

You can complete the retrieval step:
```bash
python retrieve/run.py --config `retrieve/configs/{config name}`
```


# Enabling Large Language Models to Generate Text with Citations

<p align="center"><img src="https://github.com/princeton-nlp/ALCE/blob/main/assets/moose.png?raw=true" alt="ALCE" width="15%"><br>*: ALCE is pronounced as /elk/ as ALCE is the Latin word for elk (Europe) or moose (North America).
</p>



This repository contains the code and data for paper [Enabling Large Language Models to Generate Text with Citations](https://arxiv.org/abs/2305.14627). 
In this paper, we propose ALCE, a benchmark for **A**utomatic **L**LMs' **C**itation Evaluation. 
ALCE contains three datasets: ASQA, QAMPARI, and ELI5.
We provide automatic evaluation code of LLM generations around three dimensions: fluency, correctness, and citation quality. 
This repository also includes code to reproduce the baselines in our paper.



<img src="https://github.com/princeton-nlp/ALCE/blob/main/assets/ALCE.png?raw=true" alt="ALCE" width="100%">


## Reproducing Baselines


You can reproduce baselines from our paper by 

```bash
python run.py --config configs/{config_name}
```

You can also overwrite any arguments in the config file or add new arguments simply through command line:
```
python run.py --config configs/{config_name} --seed 43 --model vicuna-13b
```

The naming of config files follow the rule of `{LLM}_{#demos and #passages}_{retriever}_{method}.yaml`. Method names include:
* `default` corresponds to the **Vanilla** model in our paper.
* `summary` corresponds to the **Summary** model.
* `extraction` corresponds to the **Snippet** model. 
* `interact_doc_id` corresponds to the **Interact** model.
* `interact_search` corresponds to the **InlineSearch** model.
* `closedbook` corresponds to the **ClosedBook** model.

Our code support both OpenAI API and offline HuggingFace models:

* For OpenAI models (for example, ChatGPT), you need to set the environment variable `OPENAI_API_KEY` and `OPENAI_ORG_ID`. If you are using the Azure OpenAI API, you need to set the environment variable of `OPENAI_API_KEY` and `OPENAI_API_BASE`. You also need to add the flag `--azure`. 
    * Note that in Azure OpenAI API, ChatGPT's name is different and you should set it by `--model gpt-35-turbo`. 
* For the open-source models, you should set the model name equal to the input of HuggingFace models' `.from_pretrained` method. This could either be a local directory (e.g. for the older LLaMA models) or a path to the HuggingFace hub. 

For detailed argument usage, please refer to `run.py`.

Model output along with gold answers and run configs will be stored in a json file in `result/`.


### Post-hoc citation

For closed-book models, one can use `post_hoc_cite.py` to add citations in a post-hoc manner (using GTR-large). To run post-hoc citation, execute
```bash
python post_hoc_cite.py --f result/{RESULT JSON FILE NAME} --external_docs data/{CORRESPONDING DATA}
```

The output file with post-hoc citations will be stored in `result/`, with a suffix `post_hoc_cite.gtr-t5-large-external`.

## Evaluation

ACLE evaluation is implemented in `eval.py`. 

For ASQA, use the following command
```bash
python eval.py --f {path/to/result/file} --citations --qa --mauve
```

For QAMPARI, use the following command
```bash
python eval.py --f {path/to/result/file} --citations
```

For ELI5, use the following command
```bash
python eval.py --f {path/to/result/file} --citations --claims_nli --mauve
```

The evaluation result will be saved in `result/`, with the same name as the input and a suffix `.score`.

## Human Evaluation

The results from our human evaluation (Section 6) are located under the directory [`human_eval`](human_eval). 
Both the data and the analysis are available, please refer to the directory for details. 

## Bug or Questions?

If you have any questions related to the code or the paper, feel free to email Tianyu (`tianyug@cs.princeton.edu`). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!



## Citation

Please cite our paper if you use ALCE in your work:

```bibtex
@inproceedings{gao2023enabling,
   title={Enabling Large Language Models to Generate Text with Citations},
   author={Gao, Tianyu and Yen, Howard and Yu, Jiatong and Chen, Danqi},
   year={2023},
   booktitle={Empirical Methods in Natural Language Processing (EMNLP)},
}
```
