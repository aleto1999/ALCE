import argparse
import csv
import json
import os
import time
import pickle

import numpy as np 
import torch
import tqdm
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# for colbert
from colbert import Indexer
from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher
import pdb

# svs
import gc
import glob
import json
import linecache
import logging
import pysvs


from typing import Literal

TOPK = 100

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


# def gtr_build_index(encoder, docs):
#     with torch.inference_mode():
#         embs = encoder.encode(docs, batch_size=4, show_progress_bar=True, normalize_embeddings=True)
#         embs = embs.astype("float16")

#     GTR_EMB = os.environ.get("GTR_EMB")
#     with open(GTR_EMB, "wb") as f:
#         pickle.dump(embs, f)
#     return embs


def gtr_wiki_retrieval(data):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("loading GTR encoder...")
    encoder = SentenceTransformer("sentence-transformers/gtr-t5-xxl", device = device)

    # encode all queries
    questions = [d["question"] for d in data]
    with torch.inference_mode():
        queries = encoder.encode(questions, batch_size=4, show_progress_bar=True, normalize_embeddings=True)
        queries = torch.tensor(queries, dtype=torch.float16, device="cpu")

    ## NOTE: this is for indexing a dataset w/ GPR
    ## for now, we are just downloading the indices directly
    # # the wikipedia split from DPR repo: https://github.com/facebookresearch/DPR
    # DPR_WIKI_TSV = os.environ.get("DPR_WIKI_TSV")
    docs = []
    # print("loading wikipedia file...")
    # with open(DPR_WIKI_TSV) as f:
    #     reader = csv.reader(f, delimiter="\t")
    #     for i, row in enumerate(reader):
    #         if i == 0:
    #             continue
    #         docs.append(row[2] + "\n" + row[1])

    GTR_EMB = os.environ.get("GTR_EMB")
    if not os.path.exists(GTR_EMB):  # check if env var set
        print("gtr embeddings not yet downloaded (see README) and cannot build withou dpr...")
        exit()
        # print("gtr embeddings not found, building...")
        # embs = gtr_build_index(encoder, docs)
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
        data[qi]["docs"] = ret
        q = q.to("cpu")

def colbert_retrieval(data):

    exp_name = 'colbert'
    with Run().context(
        RunConfig(
            nranks=1,
            experiment=exp_name,
            index_root=os.path.join(args.index_dir, 'colbert'),
        )
    ):
        # path to this file should be set :)
        name = f"COLBERT_{dataset.upper()}_EMB"
        EMB = os.environ.get(name) 
        config = ColBERTConfig(
            index_path = EMB,
            nbits=2,
            root=args.prediction_dir,  # ???  
        )
        print(f"CONFIG: {config}")
        
        # check if indices are already generated
        if not os.path.exists(EMB):
            print(f"colbert embeddings for {dataset} not found, building...")
            print('STARTING INDEXING')
            indexer = Indexer(checkpoint=os.path.join(args.model_dir, 'colbertv2.0/'), config=config)
            indexer.index(name=name, collection=EMB, overwrite = 'resume')  #'reuse')  # VAV CHANGED BECAUSE GOT INTERRUPTED?
            print('DONE INDEXING')
        else:
            print(f"colbert embeddings for {dataset} found, loading...")
            # pulled from config file in next step

        searcher = Searcher(index=name, config=config)

        # TODO: fix this to read from ALCE type file
        query_file = os.path.join(args.data_dir, args.dataset + '-queries.tsv')
        print('loading query from', query_file)
        queries = Queries(query_file)
        print('START SEARCHING')
        ranking = searcher.search_all(queries, k=100)

        # instead, use alce file format
        pred_dir = os.path.join(args.prediction_dir, exp_name)
        os.makedirs(pred_dir, exist_ok = True)

        # result_file = os.path.join(pred_dir, f'{args.dataset}.jsonl')
        # print('writing search results in', result_file)
        docs = []
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

            data[r]["docs"] = ret  # double check I'm putting this in the right spot
            q = q.to("cpu")

        print('DONE')


def svs_retrieval(data, logger):

    """
    Evaluates SVS retriever, assuming the corpus of vector embeddings have already been created
    """
    index_path = f'{args.corpus_dir}/index-{args.exp_name}'
    logger.info('STARTING INDEXING')
    try:
        # NOTE: dataloader had an error unless using the exact file path & dtype. File an issue?
        vec_loader = pysvs.VectorDataLoader(args.embed_file, pysvs.float32)
        logger.info("Vectors loaded with VectorDataLoader")
        index = args.index_fn(vec_loader, args.dist_type, **args.index_kwargs)
    except Exception as e:
        logger.info(f"Vector loader didn't work: {e}")
        vecs = pysvs.read_vecs(args.embed_file)
        logger.info(f"But direct loading of vectors did work.")
        index = args.index_fn(vecs, args.dist_type, **args.index_kwargs)
        del vecs
    try:
        index.save(index_path, index)
        logger.info(f"Saved index to {index_path}")
    except Exception as e:
        logger.info(e)
        logger.warn("Could not save index!")
    logger.info('DONE INDEXING')
  

    logger.info('EMBEDDING QUERIES')
    # Read in query data
    with open(f'{args.data_dir}/{args.dataset}.jsonl') as f:
        raw_query_data = f.readlines()
    q_ids, queries = [], []
    for rqd in raw_query_data:
        d = json.loads(rqd)
        q_ids.append(d["id"])
        queries.append(d["input"])
    # Embed queries
    if args.embed_model_type == 'st':
        import sentence_transformers as st
        embed_model = st.SentenceTransformer(args.embed_model_name)
        query_embs = embed_model.encode(queries)
    else:
        raise NotImplementedError
    # temporarily save queries to file?
    # pysvs.write_vecs(query_embs, f'{args.dataset}_embeds')

    # searcher = Searcher(index=args.corpus, config=config)
    # query_file = os.path.join(args.data_dir, args.dataset + '-queries.tsv')
    # logger.info('loading query from', query_file)
    # queries = Queries(query_file)
    logger.info('START SEARCHING')
    # ranking = searcher.search_all(queries, k=100)
    k_neighbors, dist_neighbors = index.search(query_embs, args.k)
    logger.info('DONE SEARCHING')
    # Save direct search outputs before writing to JSON
    with open('tmp.pkl', 'wb') as ftmp:
        pickle.dump([k_neighbors, dist_neighbors], ftmp)
    logger.info(f'Dumped neighbors to {ftmp}!')
    del index, vec_loader
    gc.collect() 

    # Handle output directories
    pred_dir = os.path.join(args.prediction_dir, args.exp_name)
    os.makedirs(pred_dir, exist_ok=True)
    result_file = os.path.join(pred_dir, f'{args.dataset}.jsonl')
    logger.info(f'writing search results in {result_file}')

    # # Load id2title mapping for results writing
    # with open(f'{args.corpus_dir}/{args.corpus}/id2title.json', 'rb') as f:
    #     lines = f.readlines()
    # tmp_dict = [json.loads(l) for l in lines]
    # idict = dict()
    # for d in tmp_dict:
    #     idict.update({d["id"]: d["title"]})
    # del lines, tmp_dict

    cfiles = glob.glob(f'{args.corpus_dir}/{args.corpus}_jsonl/*.jsonl')
    corpus_file = cfiles[0]
    logger.info(f"Corpus is in {corpus_file}")
    empty_dict = {
        "score": '',
        "text": '',
        "page_id": '',
        "start_par_id": '',
        "end_par_id": ''
    }

    with open(corpus_file, 'r') as f:
        clines = f.readlines()
    with open(result_file, 'w') as output_file:
        for i, qq in enumerate(tqdm.tqdm(q_ids)):
            result = {}
            result['id'] = q_ids[i]
            result['input'] = queries[i]
            result['output'] = [{}]  # Initialize
            # Get provenance information from all k neighbors
            neighbor_ids = k_neighbors[i, :]
            # Just read file from the line corresponding to the index of the neighbor
            try:
                wp_dicts = [json.loads(clines[int(n)]) if n != 0 else None for n in
                            neighbor_ids]
            except Exception as e:
                logger.info(e)
                import pdb; pdb.set_trace()
                
            provenance = [
                {"score": str(dist_neighbors[i, j]),
                 "text": wp_dicts[j][args.data_key],
                 "page_id": wp_dicts[j]["id"].split("_")[0],
                 "start_par_id": wp_dicts[j]["id"].split("_")[1],
                 "end_par_id": wp_dicts[j]["id"].split("_")[1]} \
                 if wp_dicts[j] \
                    else empty_dict \
                        for j in range(args.k)
            ]
                    # "wikipedia_id": wp_dicts[j]["id"].split("_")[0],
                    # "wikipedia_title": idict[wp_dicts[j]["id"].split("_")[0]]} for j in range(args.k)]

            # write line in output file
            result['output'][0]['provenance'] = provenance
            json.dump(result, output_file)
            output_file.write('\n')

    logger.info('DONE')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Passage retrieval.")
    parser.add_argument("--retriever", type=str, default=None, help="options: bm25/gtr/colbert")
    parser.add_argument("--data_file", type=str, default=None, help="path to the data file with queries")
    parser.add_argument("--dataset", type=str, default=None, help="options:qampari/// ; required for colbert")
    parser.add_argument("--output_file", type=str, default=None, help="same format as the data file but with the retrieved docs.")
    args = parser.parse_args()

    with open(args.data_file) as f:
        data = json.load(f)

    if args.retriever == "bm25":
        bm25_sphere_retrieval(data)
    elif args.retriever == "gtr":
        gtr_wiki_retrieval(data)
    elif args.retriever == "colbert":
        if args.dataset in ["qampari"]:
            dataset = args.dataset
        else:
            raise Exception  # TODO
        
        colbert_retrieval(data, dataset)
    else:
        raise NotImplementedError

    with open(args.output_file, "w") as f:
        json.dump(data, f, indent=4)
