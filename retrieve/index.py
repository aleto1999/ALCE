import torch
import os
import pickle

def gtr_build_index(
    index_path,
    encoder,
    doc_dataset,
    docs,
    logger
):
    """
    Builds gtr index and dumps into a pickle in index directory
    """
    with torch.inference_mode():
        embs = encoder.encode(docs, batch_size=4, show_progress_bar=True, normalize_embeddings=True)
        embs = embs.astype("float16")

    index_file = os.path.join(index_path, f"{doc_dataset}.pkl")
    with open(index_file, "wb") as f:
        pickle.dump(embs, f)  # do i want to stick with pickles??
    return embs


def colbert_build_index(
        config,
        model_dir,
        doc_dataset,
        doc_dataset_path,
        logger
):
    """
    Builds colbert index with colbertv2.0 for doc dataset
    """
    from colbert import Indexer

    indexer = Indexer(
        checkpoint=os.path.join(model_dir, 'colbertv2.0/'),
        config=config
    )
    indexer.index(
        name=doc_dataset,
        collection=doc_dataset_path,
        overwrite = 'resume'  #'reuse'
    )


def svs_build_index(
    index_path: str,
    vec_file: str,
    index_fn,
    dist_type,
    index_kwargs,
    logger
):
    """
    Builds svs index assuming doc dataset vectors have been generated
    """
    import pysvs

    try:
        # NOTE: dataloader had an error unless using the exact file path & dtype. File an issue?
        vec_loader = pysvs.VectorDataLoader(vec_file, pysvs.float32)
        logger.info("Vectors loaded with VectorDataLoader")
        index = index_fn(vec_loader, dist_type, **index_kwargs)
    except Exception as e:
        logger.info(f"Vector loader didn't work: {e}")
        vecs = pysvs.read_vecs(vec_file)
        logger.info(f"But direct loading of vectors did work.")
        index = index_fn(vecs, dist_type, **index_kwargs)
        del vecs
    try:
        index.save(index_path, index)
        logger.info(f"Saved index to {index_path}")
    except Exception as e:
        logger.info(e)
        logger.warn("Could not save index!")