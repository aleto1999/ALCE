import argparse
import csv
import json
import os
import time
import pickle
import logging

import retrieval.index as index

import numpy as np 
import torch
import tqdm
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# for colbert
from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher
from colbert import Indexer
import pdb

# svs
import gc
import glob
import json
import linecache
import logging
import pysvs

from typing import Literal

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
TOPK = 100

class InvalidArgument(Exception):
    """raise when user input arguments are invalid"""
    pass

class IncompleteSetup(Exception):
    """raise when user did not complete environmnet variable setup (see README and setup.sh)"""
    pass

class DocDataset():

    def __init__(self, name: str, path:str=None): 
        self.name = name.lower()
        self.path = path
        self.docs = None
        self.load_data()

    def load_data(self):
        if self.name == "dpr_wiki":
            DPR_WIKI_TSV = os.environ.get("DPR_WIKI_TSV")
            docs = []
            print("loading wikipedia file...")
            with open(DPR_WIKI_TSV) as f:
                reader = csv.reader(f, delimiter="\t")
                for i, row in enumerate(reader):
                    if i == 0:
                        continue
                    docs.append(row[2] + "\n" + row[1])
            self.docs = docs
        elif self.name == "qampari":
            self.path = ""  # path to tsv

        else:
            print(f"not loading data in doc dataset class (not yet implemented for {self.name})")


def bm25_sphere_retrieval(data):
    from pyserini.search import LuceneSearcher
    index_path = os.environ.get("BM25_SPHERE_PATH")
    print("loading bm25 index, this may take a while...")
    searcher = LuceneSearcher(index_path)

    print("running bm25 retrieval...")
    for d in tqdm(data):
        query = d["question"]
        try:
            hits = searcher.search(query, TOPK)
        except Exception as e:
            #https://github.com/castorini/pyserini/blob/1bc0bc11da919c20b4738fccc020eee1704369eb/scripts/kilt/anserini_retriever.py#L100
            if "maxClauseCount" in str(e):
                query = " ".join(query.split())[:950]
                hits = searcher.search(query, TOPK)
            else:
                raise e

        docs = []
        for hit in hits:
            h = json.loads(str(hit.docid).strip())
            docs.append({
                "title": h["title"],
                "text": hit.raw,
                "url": h["url"],
            })
        d["docs"] = docs

    return data

def gtr_wiki_retrieval(query_data, docs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("loading GTR encoder...")
    encoder = SentenceTransformer("sentence-transformers/gtr-t5-xxl", device = device)

    # encode all queries
    questions = [d["question"] for d in query_data]
    with torch.inference_mode():
        queries = encoder.encode(questions, batch_size=4, show_progress_bar=True, normalize_embeddings=True)
        queries = torch.tensor(queries, dtype=torch.float16, device="cpu")

    ## NOTE: this is for indexing a dataset w/ GPR
    ## for now, we are just downloading the indices directly
    # # the wikipedia split from DPR repo: https://github.com/facebookresearch/DPR
    docs = docs.docs

    GTR_EMB = os.environ.get("GTR_EMB")
    if not os.path.exists(GTR_EMB):  # check if env var set
        print("gtr embeddings not found, building...")
        embs = index.gtr_build_index(encoder, docs)
    else:
        print("gtr embeddings found, loading...")
        with open(GTR_EMB, "rb") as f:
            embs = pickle.load(f)

    del(encoder) # save gpu mem

    gtr_emb = torch.tensor(embs, dtype=torch.float16, device=device)

    print("running GTR retrieval...")
    for qi, q in enumerate(tqdm(queries)):
        q = q.to(device)
        scores = torch.matmul(gtr_emb, q)
        score, idx = torch.topk(scores, TOPK)
        ret = []
        for i in range(idx.size(0)):
            title, text = docs[idx[i].item()].split("\n")
            ret.append({"id": str(idx[i].item()+1),"title": title, "text": text, "score": score[i].item()})
        query_data[qi]["docs"] = ret
        q = q.to("cpu")

    return query_data


def colbert_retrieval(query_data, dataset):

    COLBERT_IN_PATH = os.environ.get("COLBERT_IN_PATH")
    MODEL_PATH = os.path.join(  # path to model used for indexing
        COLBERT_IN_PATH,
        'colbertv2.0/'
    )
    QUERY_PATH = os.path.join(  # path to input query tsv 
        COLBERT_IN_PATH,
        f"{args.dataset}-queries.tsv"
    )

    INDEX_PATH = os.environ.get("INDEX_PATH")
    index_ret_path = os.path.join(INDEX_PATH, 'colbert')

    exp_name = 'colbert'
    prediction_temp_dir = ""  # TODO

    with Run().context(
        RunConfig(
            nranks=1,
            experiment=exp_name,
            index_root=index_ret_path,
        )
    ):
        index_ds_path = os.path.join(index_ret_path, dataset.name)
        config = ColBERTConfig(
            index_path = index_ds_path,
            nbits=2,
            root=prediction_temp_dir,
        )
        logger.info(f"Config file for Colbert retrieval: {config}")
        
    
        logger.info("Start indexing....")
        index.colbert_build_index(
            config,
            MODEL_PATH,
            dataset
        )
        logger.info("Indexing complete")

        searcher = Searcher(index=dataset.name, config=config)
        
        logger.info(f"Loading queries from: {QUERY_PATH}")
        queries = Queries(QUERY_PATH)

        logger.info("Starting search...")
        ranking = searcher.search_all(queries, k=100)

        for r_id, r in enumerate(ranking.data):
            ret = []
            for i, (c_id, rank, score) in enumerate(ranking.data[r]):
                page_id, paragraph_id = searcher.collection.pid_list[c_id].split('_')
                ret.append({
                    "id": page_id,
                    "start_par_id": paragraph_id,
                    "end_par_id": paragraph_id,
                    "title": None,  # need this?
                    "text": searcher.collection.data[c_id],
                    "score": score
                })

            query_data[r]["docs"] = ret  # double check I'm putting this in the right spot
            q = q.to("cpu")

        logger.info("Search complete")

    return query_data


def svs_retrieval(query_data, embedding_name, index_fn, logger):

    """
    Evaluates SVS retriever, assuming the corpus of vector embeddings have already been created
    """
    index_path = f'{args.corpus_dir}/index-{args.exp_name}'
    INDEX_PATH = os.environ.get("INDEX_PATH")
    index_path = os.path.join(INDEX_PATH, 'svs')

    VEC_PATH = os.environ.get("VEC_PATH")
    vec_file = os.path.join(VEC_PATH, embedding_name)

    logger.info('Start indexing...')
    index.svs_build_index(
        index_path,
        vec_file,
        index_fn,
        logger
    )
    logger.info('Done indexing')

    logger.info('Embedding queries...')
    queries = [d["question"] for d in query_data]
    if embed_model_type == 'st':
        import sentence_transformers as st
        embed_model = st.SentenceTransformer(args.embed_model_name)
        query_embs = embed_model.encode(queries)
    else:
        raise NotImplementedError

    logger.info('Start searching...')
    k_neighbors, dist_neighbors = index.search(query_embs, args.k)
    logger.info('Done searching')

    # Save direct search outputs before writing to JSON
    # TODO: do i really need this?
    with open('tmp.pkl', 'wb') as ftmp:
        pickle.dump([k_neighbors, dist_neighbors], ftmp)
    logger.info(f'Dumped neighbors to {ftmp}!')
    del index, vec_loader
    gc.collect() 
    
    # TODO: wth does this do??s
    cfiles = glob.glob(f'{args.corpus_dir}/{args.corpus}_jsonl/*.jsonl')
    corpus_file = cfiles[0]
    logger.info(f"Corpus is in {corpus_file}")
    with open(corpus_file, 'r') as f:
        clines = f.readlines()
    empty_dict = {
        "id": "",
        "title": "",
        "text": "",
        "score": ""
    }
    for qi, q in enumerate(tqdm(queries)):
        neighbor_ids = k_neighbors[qi, :]
        try:
            wp_dicts = [json.loads(clines[int(n)]) if n != 0 else None for n in neighbor_ids]
        except Exception as e:
            logger.info(e)
            import pdb; pdb.set_trace()

        ret = \
        [
            {
                "id": -1,  # TODO
                "title": "",
                "text": wp_dicts[j]["text"],
                "score": str(dist_neighbors[qi, j])
            } if wp_dicts[j] else empty_dict for j in range(TOPK)
        ]
        query_data[qi]['docs'] = ret

    logger.info('DONE')


def main(args):

    # check that index file path env variable has been set 
    INDEX_PATH = os.environ.get("INDEX_PATH")
    if not os.path.exists(INDEX_PATH):
        print("Please set environment variable $INDEX_PATH to specify where to save or load indices.")
        raise IncompleteSetup
    
    # check that query file exists
    if not os.path.exists(args.query_data_file):
        print(f"Query file specified does not exist: {args.query_data_file}")
        raise InvalidArgument
    else:
        # open specified data file (containing queries)
        # will append retrieved docs to this file 
        with open(args.data_file) as f:
            query_data = json.load(f)


    if args.retriever == "bm25":
        bm25_sphere_retrieval(query_data)

    elif args.retriever == "gtr":
        gtr_wiki_retrieval(query_data)

    elif args.retriever == "colbert":
        if args.dataset in ["qampari"]:
            doc_dataset = DocDataset(
                name=args.dataset
            )
        data = colbert_retrieval(query_data, doc_dataset)

    elif args.retriever == "svs":
        if args.dataset in ["qampari"]:
            doc_dataset = DocDataset(
                name=args.dataset
        )
        data = svs_retrieval(query_data, doc_dataset)

    else:
        print(f"Invalid retriever: {args.retriever}")
        print("Current implemented options include: bm25/gtr/colbert/svs")
        raise InvalidArgument

    with open(args.output_file, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Passage retrieval.")
    parser.add_argument("--retriever", type=str, default=None, help="options: bm25/gtr/colbert/svs")
    parser.add_argument("--data_file", type=str, default=None, help="path to the data file with queries")
    parser.add_argument("--dataset", type=str, default=None, help="options:qampari/// ; required for colbert")
    parser.add_argument("--output_file", type=str, default=None, help="same format as the data file but with the retrieved docs.")
    args = parser.parse_args()

    main(args)
