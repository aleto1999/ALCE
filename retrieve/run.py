import argparse
import json
import os
import pickle
import logging
import yaml
import numpy as np 
import torch
import gc
import tqdm
from tqdm import tqdm

# import indexing functions for all retrievers
import retrieve.index as index

class InvalidArgument(Exception):
    """raise when user input arguments are invalid"""
    pass

class IncompleteSetup(Exception):
    """raise when user did not complete environmnet variable setup (see README and setup.sh)"""
    pass

DATASET_PATH = os.environ.get("DATASET_PATH")
COLBERT_MODEL_PATH = os.environ.get("COLBERT_MODEL_PATH")
INDEX_PATH = os.environ.get("INDEX_PATH")
VEC_PATH = os.environ.get("VEC_PATH")

def colbert_retrieval(
    query_data,
    query_dataset,
    doc_dataset,
    logger
):
    """
    Index and retrieve with ColBERT
    """

    from colbert.data import Queries
    from colbert.infra import Run, RunConfig, ColBERTConfig
    from colbert import Searcher


    MODEL_PATH = os.path.join(  # path to model used for indexing
        COLBERT_MODEL_PATH,
        'colbertv2.0/'
    )
    QUERY_PATH = os.path.join(  # path to input query tsv 
        DATASET_PATH,
        query_dataset,
        "queries.tsv"
    )
    DOC_PATH = os.path.join(
        DATASET_PATH,
        query_dataset,
        "docs.tsv"
    )

    index_ret_path = os.path.join(INDEX_PATH, 'colbert')

    exp_name = f'colbert'
    prediction_temp_dir = ""  # TODO

    with Run().context(
        RunConfig(
            nranks=1,
            experiment=exp_name,
            index_root=index_ret_path,
        )
    ):
        # index documents in $INDEX_DIR/colbert/doc_dataset
        index_ds_path = os.path.join(index_ret_path, doc_dataset)
        config = ColBERTConfig(
            index_path=index_ds_path,
            nbits=2,
            root=prediction_temp_dir,
        )
        logger.info(f"Config for Colbert retrieval: {config}")
        
    
        logger.info("Start indexing....")
        index.colbert_build_index(
            config,
            MODEL_PATH,
            doc_dataset,
            DOC_PATH,
            logger
        )
        logger.info("Indexing complete")

        searcher = Searcher(index=doc_dataset, config=config)
        
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

 def bge_retrieval(
    retriever: str  # bge-base or bge-large
):
    raise NotImplementedError


def svs_retrieval(
    query_data: dict,
    k: int,
    doc_dataset: str,
    embed_file: str,
    embed_model_name: str,
    embed_model_type: str,
    index_fn,
    dist_type,
    index_kwargs,
    logger
):

    """
    SVS retrieval assuming the corpus of vector embeddings have already been created
    """

    index_path = os.path.join(INDEX_PATH, 'svs', doc_dataset)  # how to save these?
    vec_file = os.path.join(VEC_PATH, embed_file)

    logger.info('Start indexing...')
    index.svs_build_index(
        index_path,
        vec_file,
        index_fn,
        dist_type,
        index_kwargs,
        logger
    )
    logger.info('Done indexing')

    logger.info('Embedding queries...')
    queries = [d["question"] for d in query_data]
    if embed_model_type == 'st':
        import sentence_transformers as st
        embed_model = st.SentenceTransformer(embed_model_name)
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
    
    # I really need to fix this
    corpus_file = os.path.join(DATASET_PATH, doc_dataset, "docs.jsonl")
    with open(corpus_file, 'r') as f:
        clines = f.readlines()

    empty_dict = {
        "id": "",
        "start_par_id": "",
        "end_par_id": "",
        "title": "",
        "text": "",
        "score": ""
    }
    for qi, q in enumerate(tqdm(queries)):
        neighbor_ids = k_neighbors[qi, :]
        try:
            # read lines from json corresponding to neighbors
            wp_dicts = [json.loads(clines[int(n)]) if n != 0 else None for n in neighbor_ids]
        except Exception as e:
            logger.info(e)
            import pdb; pdb.set_trace()

        # loop over neighbor dictionaries (pulled from json)
        ret = \
        [
            {
                "id": wp_dicts[j]["id"].split("_")[0],
                "start_par_id": wp_dicts[j]["id"].split("_")[1],
                "end_par_id": wp_dicts[j]["id"].split("_")[1],
                "title": None,  #fix this?
                "text": wp_dicts[j]["text"],
                "score": str(dist_neighbors[qi, j])

            } if wp_dicts[j] else empty_dict for j in range(k)
        ]
        query_data[qi]['docs'] = ret

    logger.info('DONE')


def main(args):

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # check that index file path env variable has been set 
    INDEX_PATH = os.environ.get("INDEX_PATH")
    if not os.path.exists(INDEX_PATH):
        logger.info("Please set environment variable $INDEX_PATH to specify where to save or load indices.")
        raise IncompleteSetup
    
    # check that query file exists
    if not os.path.exists(args.query_data_file):
        logger.info(f"Query file specified does not exist: {args.query_data_file}")
        raise InvalidArgument
    else:
        # open specified data file (containing queries)
        # will append retrieved docs to this file 
        with open(args.data_file) as f:
            query_data = json.load(f)


    if args.retriever == "colbert":

        if not args.query_dataset in ["nq"]:
            logger.info(f"query dataset {args.query_dataset} not implemented for colbert")
            raise InvalidArgument
        if not args.doc_dataset in ["kilt_wikipedia"]:
            logger.info(f"doc dataset {args.doc_dataset} not implemented for colbert")
            raise InvalidArgument

        data = colbert_retrieval(
            query_data,
            args.query_dataset,
            args.doc_dataset,
            logger
        )

    elif "bge" in args.retriever: # bge-large or bge-base
        raise NotImplementedError
        data = bge_retrieval(args.retriever)

    elif args.retriever == "svs":
        # check that all required arguments are specified
        if not args.embed_model_type:
            logger.info("model_type must be specified for svs")
            raise InvalidArgument
        if not args.embedding_name:
            logger.info("embedding_name must be specified for svs")
            raise InvalidArgument
        if not args.index_fn:
            logger.info("index_fn must be specified for svs")
            raise InvalidArgument

        data = svs_retrieval(
            query_data,
            args.k,
            args.dataset,
            args.embed_file,
            args.embed_model_name,
            args.embed_model_type,
            args.index_fn,
            args.dist_type,
            args.index_kwargs,
            logger
        )

    else:
        print(f"Invalid retriever: {args.retriever}")
        print("Current implemented options include: colbert/svs")
        raise InvalidArgument

    with open(args.output_file, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Passage retrieval.")
    parser.add_argument("--config", type=str, default=None, help="Path to the config file")
    parser.add_argument("--k", type=int, default=100, help="Number of nearest neighbors to retrieve")
    parser.add_argument("--retriever", type=str, required=True, help="options: bm25/gtr/colbert/svs")
    parser.add_argument("--query_file", type=str, required=True, help="path to the data file with queries")
    parser.add_argument("--query_dataset", type=str, required=True, help="query dataset name -> options: nq")
    parser.add_argument("--doc_dataset", type=str, required=True, help="dataset name -> options: kilt_wikipedia/// ")
    parser.add_argument("--output_file", type=str, default=None, help="Same format as the data file but with the retrieved docs; if not specified, will be added to the original data_file")
 
    # svs arguments
    parser.add_argument("--embed_file", default=None, help='path to .fvecs vector embedding file from $VEC_PATH')
    parser.add_argument("--embed_model_name", default=None, help='model to use for embedding queries with SentenceTransformer (eg. "snowflake/snowflake-arctic-embed-s")')
    parser.add_argument("--embed_model_type", default="st", help="Type of embedding model to use, choose from [st, hf]. st is SentenceTransformers and hf is HuggingFace")
    parser.add_argument("--index_fn", default=pysvs.Flat, help='type of pysvs index')
    parser.add_argument("--dist_type", default=pysvs.DistanceType.MIP, help='type of distance to use for index/search')
    parser.add_argument("--index_kwargs", type=json.loads, default='{}', help='additional input arguments for index building')
    args = parser.parse_args()

    # get defaults from config file
    config = yaml.safe_load(open(args.config)) if args.config is not None else {}
    parser.set_defaults(**config)

    args = parser.parse_args()

    main(args)
