import os
import sys

sys.path.insert(0, "../third_party/ColBERT")

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
from colbert import Indexer

DATAROOT = "/mnt/"
DATASET = "enw100"
NUM_GPUS = 8


def generate_index(collection):
    nbits = 1
    doc_maxlen = 150

    checkpoint = "/mnt/workspace/downloads/colbertv2.0"
    index_name = f"{DATASET}.{nbits}bits"

    with Run().context(RunConfig(nranks=NUM_GPUS, experiment="enwiki")):
        config = ColBERTConfig(doc_maxlen=doc_maxlen, nbits=nbits)

        indexer = Indexer(checkpoint=checkpoint, config=config)
        indexer.index(name=index_name, collection=collection, overwrite=True)
    print(f"Index at: {indexer.get_index()}")


if __name__ == "__main__":
    collection_path = os.path.join(DATAROOT, DATASET, "en_w100.tsv")
    collection = Collection(path=collection_path)
    print(f"Loaded {len(collection):,} passages")

    generate_index(collection)
