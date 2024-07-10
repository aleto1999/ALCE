import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import argparse
import os
# import openai
import json
from tqdm import tqdm
from transformers import AutoTokenizer
import time
import string
import numpy as np
import re
from searcher import SearcherWithinDocs
import yaml
from utils import *
from nltk import sent_tokenize

def remove_citations(sent):
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")

class LLM:

    def __init__(self, args):
        self.args = args
        self.model, self.tokenizer = self.load_model(args.model)
        
        self.prompt_exceed_max_length = 0
        self.fewer_than_50 = 0
        self.azure_filter_fail = 0

    def load_model(self, model_name_or_path, dtype=torch.float16, int8=False, reserve_memory=10):
        # Load a huggingface model and tokenizer
        # dtype: torch.float16 or torch.bfloat16
        # int8: whether to use int8 quantization
        # reserve_memory: how much memory to reserve for the model on each gpu (in GB)

        # Load the FP16 model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        logger.info(f"Loading {model_name_or_path} in {dtype}...")
        if int8:
            logger.warn("Use LLM.int8")
        start_time = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map='auto',
            torch_dtype=dtype,
            max_memory=get_max_memory(),
            load_in_8bit=int8,
        )
        print(f"Model running on device: {model.device}")
        logger.info("Finish loading in %.2f sec." % (time.time() - start_time))

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)

        # Fix OPT bos token problem in HF
        if "opt" in model_name_or_path:
            tokenizer.bos_token = "<s>"
        tokenizer.padding_side = "left"

        return model, tokenizer


    def generate(self, prompt, max_tokens, stop=None):
        args = self.args
        if max_tokens <= 0:
            self.prompt_exceed_max_length += 1
            logger.warning("Prompt exceeds max length and return an empty string as answer. If this happens too many times, it is suggested to make the prompt shorter")
            return ""
        if max_tokens < 50:
            self.fewer_than_50 += 1
            logger.warning("The model can at most generate < 50 tokens. If this happens too many times, it is suggested to make the prompt shorter")

        
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        stop = [] if stop is None else stop
        stop = list(set(stop + ["\n", "Ċ", "ĊĊ", "<0x0A>"])) # In Llama \n is <0x0A>; In OPT \n is Ċ
        stop_token_ids = list(set([self.tokenizer.convert_tokens_to_ids(stop_token) for stop_token in stop] + [self.model.config.eos_token_id]))
        if "llama" in args.model.lower():
            stop_token_ids.remove(self.tokenizer.unk_token_id)

        outputs = self.model.generate(
            **inputs,
            do_sample=True, temperature=args.temperature, top_p=args.top_p, 
            max_new_tokens=max_tokens,
            num_return_sequences=1,
            eos_token_id=stop_token_ids
        )
        generation = self.tokenizer.decode(outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=True)
        return generation

def save_results(
    args,
    eval_data
):
    # Save the result
    model_name = args.model
    if "/" in model_name:
        model_name = model_name.split("/")[-1]
    name = f"{args.dataset_name}-{model_name}-{args.tag}-shot{args.shot}-ndoc{args.ndoc}-{args.seed}"
    if args.azure:
        name += "-azure"
    if args.quick_test is not None:
        name += f"-quick_test{args.quick_test}"
    if args.no_doc_in_demo:
        name += "-no_doc_in_demo"
    if args.fewer_doc_in_demo:
        name += f"-{args.ndoc_in_demo}_doc_in_demo"
    if args.num_samples > 1:
        name += f"-sample{args.num_samples}"
    if args.force_cite_show:
        name += f"-forceciteshow"

    eval_data = {
        "args": args.__dict__,
        "data": eval_data,
    }

    if not os.path.exists("result"):
        os.makedirs("result")
    json.dump(eval_data, open("result/" + name + ".json", "w"), indent=4)


def main(args):
    # Load config
    config = yaml.safe_load(open(args.config)) if args.config is not None else {}
    parser.set_defaults(**config)
    args = parser.parse_args()
    for k in args.__dict__:
        print(f"{k}: {args.__dict__[k]}")

    logger.info(f"Set the model max length to {args.max_length} (if not correct, check the code)")

    # Load the model
    llm = LLM(args)
    
    # Generate prompts
    np.random.seed(args.seed)

    # Load data
    prompt_data = json.load(open(args.prompt_file))
    eval_data = json.load(open(args.eval_file))

    # Generate the demonstration part
    head_prompt = ""
    train_ids = np.random.choice(len(prompt_data["demos"]), args.shot, replace=False)
    for train_id in train_ids:
        train_item = prompt_data["demos"][train_id]
        ndoc = args.ndoc
        if args.no_doc_in_demo:
            ndoc = 0
        elif args.fewer_doc_in_demo:
            assert args.ndoc_in_demo is not None
            ndoc = args.ndoc_in_demo
        head_prompt += make_demo(  # add 'shots'
            train_item, prompt=prompt_data["demo_prompt"], ndoc=ndoc, doc_prompt=prompt_data["doc_prompt"], 
            instruction=prompt_data["instruction"], use_shorter=args.use_shorter 
        )
        head_prompt += prompt_data["demo_sep"]

    # Sample quick test
    if args.quick_test is not None:
        eval_ids = np.random.choice(len(eval_data), args.quick_test, replace=False)
        eval_data = [eval_data[int(idx)] for idx in eval_ids]

    logger.info("Generating prompts...") 
    incomplete_doc_list = 0 # For some questions there might be fewer than ndoc documents
    for idx, eval_item in enumerate(tqdm(eval_data)):
        eval_data[idx]['prompt'] = head_prompt + make_demo(
            eval_item,
            prompt=prompt_data["demo_prompt"],
            ndoc=args.ndoc,
            doc_prompt=prompt_data["doc_prompt"],
            instruction=prompt_data["instruction"],
            use_shorter=args.use_shorter, 
            test=True
        )
        doc_list = get_shorter_text(
            eval_item,
            eval_item["docs"],
            args.ndoc,
            args.use_shorter
        ) if args.use_shorter is not None else eval_item["docs"][:args.ndoc]

        if not args.retrieve_in_all_docs:
            # If --retrieve_in_all_docs, we keep the original docs and do not trim them by ndoc
            # Otherwise, take the new docs (truncated by ndoc and filtered if using summary/extraction)
            eval_data[idx]['docs'] = doc_list
        if len(doc_list) < args.ndoc:
            incomplete_doc_list += 1
    logger.info("Done.")
    if incomplete_doc_list > 0:
        logger.warning(f"There are {incomplete_doc_list} questions that have incomplete document list (may due to a lot of them are filtered out by summary/extraction).")

    for idx, item in enumerate(tqdm(eval_data)):
        prompt = item['prompt']
        prompt_len = len(llm.tokenizer.tokenize(prompt))

        if idx == 0:
            print(prompt)

        output_array = []
        for _ in range(args.num_samples):
            output = llm.generate(
                prompt,
                min(args.max_new_tokens, args.max_length-prompt_len)
            )
            output_array.append(output)
            item['prompt'] = prompt
            
            output_array[-1] = output_array[-1].replace("<|im_end|>", "").rstrip()
            if output_array[-1].endswith("End."):
                output_array[-1] = output_array[-1][:-len("End.")]

            logger.info(f"Prompt length={prompt_len}")
            logger.info(f"Question: {item['question']}")
            logger.info(f"Gold answer: {item['answer']}")
            logger.info(f"Final model output: {output_array[-1]}") 
        
        item['output'] = output_array if len(output_array) > 1 else output_array[0]
        
    logger.info(f"#Cases when prompts exceed max length: {llm.prompt_exceed_max_length}")
    logger.info(f"#Cases when max new tokens < 50: {llm.fewer_than_50}")

    save_results(args, eval_data)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to the config file")

    # Prompt file is a json file that contains the following fields:
    # - instruction: the instruction, which will appear at the beginning of each demo and the test example
    # - demo_sep: the separator between each demo, for example, "\n\n\n"
    # - demo_prompt: the prompt for the demo, for example, "Instruction: {INST}\n\nQuestion: {Q}\n\n{D}\nAnswer: {A}"
    #     - {INST}: the instruction
    #     - {D}: the documents
    #     - {Q}: the question
    #     - {A}: the answers
    # - doc_prompt, the prompt for each document, for example, "Document [{ID}](Title: {T}): {P}", where
    #     - {ID}: the document id, staring from 1
    #     - {T}: the document title
    #     - {P}: the document text
    # - demos: a list of demo examples, each of which should have
    #     - question: the question
    #     - docs: the documents ("title" and "text")
    #     - answer: the answer to show in the demo. If it is a list, they will be concatenated by "\n". This is useful when the answer includes interactive components.
    # Note that this python file will sample `--shot` demos from the prompt file given the random seed `--seed`
    parser.add_argument("--prompt_file", type=str, help="Path to the prompt file")

    # Evaluation file is a json file that contains a list of item, each of which contains
    # - question: the question
    # - answer: the answer
    # - docs: the documents, each of which contains "title", "text"
    parser.add_argument("--eval_file", type=str, help="Path to the eval file")
    parser.add_argument("--quick_test", type=int, default=None, help="Quickly test a few examples")

    # ICL setting
    parser.add_argument("--ndoc", type=int, help="Number of documents")
    parser.add_argument("--shot", type=int, help="Number of ICL demonstrations")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the random number generator")
    parser.add_argument("--no_doc_in_demo", type=bool, default=False, help="Whether to remove the documents in the demos")
    parser.add_argument("--fewer_doc_in_demo", type=bool, default=False, help="Whether to use fewer documents in the demos")
    parser.add_argument("--ndoc_in_demo", type=int, default=None, help="When using --fewer_doc_in_demo, use this to designate how many docs in demo")

    # Model and name
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset (for saving)")
    parser.add_argument("--tag", type=str, help="Tag of run (for saving)")
    parser.add_argument("--model", type=str, help="Model to use")
    parser.add_argument("--openai_api", type=bool, default=False, help="Whether to use OpenAI API")
    parser.add_argument("--azure", action="store_true", default=False, help="Azure openai API")

    # Decoding
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for decoding")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling top-p")
    parser.add_argument("--max_new_tokens", type=int, default=300, help="Max number of new tokens to generate in one step")
    parser.add_argument("--max_length", type=int, required=True, help="Max length the model can take. Should set properly wrt the model to avoid position overflow.")
    parser.add_argument("--num_samples", type=int, default=1, help="Sample multiple answers.")

    # Use summarization/extraction of the documents
    parser.add_argument("--use_shorter", type=str, default=None, help="Whether to use summary data or extraction data for documents. Option: None, `summary`, `extraction`")

    # # Interactive
    # parser.add_argument("--interactive", type=bool, default=False, help="Whether to run in interactive mode")
    # parser.add_argument("--interactive_query", type=str, default=None, help="The query to use in interactive mode, either `doc_id` (corresponding to interact in paper) or `search` (corresponding to inlinesearch in paper).")
    # parser.add_argument("--retriever", type=str, default=None, help="When using interactive search mode, which retriever to use. Options: `tfidf`, `gtr-t5-large`")
    # parser.add_argument("--retriever_device", type=str, default="cuda", help="Where to put the dense retriever if using. Options: `cuda`, `cpu`")
    # parser.add_argument("--retrieve_in_all_docs", type=bool, default=False, help="Retrieve in all documents instead of just top ndoc")
    # parser.add_argument("--max_turn", type=int, default=10, help="Max number of all actions")
    # parser.add_argument("--max_doc_show", type=int, default=3, help="Max number of documents to show at one time.")
    # parser.add_argument("--force_cite_show", type=bool, default=False, help="Force citing the documents that are shown to the model")
    args = parser.parse_args()

    main()
