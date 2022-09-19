import os
import sys
import ast
from collections import OrderedDict
import json

sys.path.insert(0, "../third_party/ColBERT")

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries
from colbert import Searcher 

index_name = "enw100.1bits"

def load_mqueries(queries_path):
    queries = OrderedDict()

    print("#> Loading the multilingual queries from", queries_path, "...")

    with open(queries_path) as f:
        for line in f:
            qid, translated_query, query, lang, answers = line.strip().split("\t")
            # qid = int(qid)

            assert qid not in queries, ("Query QID", qid, "is repeated!")
            queries[qid] = (translated_query, query, lang, answers)

    print("#> Got", len(queries), "queries. All QIDs are unique.\n")

    return queries

def get_query_result(qid, query, rankings, searcher):
    query_result = {}
    query_result['q_id'] = qid
    query_result['question'] = query[1]
    query_result['answers'] = ast.literal_eval(query[3])
    query_result['lang'] = query[2]
    ctxs = []
    for passage_id, passage_rank, passage_score in rankings[qid]:
        ctx = {}
        ctx['id'] = passage_id
        split_passage = searcher.collection[passage_id].split(' | ', 1)
        ctx['title'] = split_passage[0]
        ctx['text'] = split_passage[1]
        ctx['score'] = passage_score
        ctx['has_answer'] = None
        ctxs.append(ctx)
    query_result['ctxs'] = ctxs
    return query_result

def main():
    queries_path = "../data/mkqa_queries_translated.tsv"
    queries = load_mqueries(queries_path)

    query_text = OrderedDict()
    for qid in queries:
        query_text[qid] = queries[qid][0]

    with Run().context(RunConfig(index_root='/workspace/index', experiment='enwiki')):
        searcher = Searcher(index=index_name)
    rankings = searcher.search_all(query_text, k=50).todict()

    output = []
    for qid in queries:
        query = queries[qid]
        output.append(get_query_result(qid, query, rankings, searcher))

    with open("../data/colbert_outputs.json", "w") as outfile:
        json.dump(output, outfile)

if __name__ == "__main__":
    main()
