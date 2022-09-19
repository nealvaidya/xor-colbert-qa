# xor-colbert-qa

## Part 1 - Index Data
Index wikipedia data using ColBERT v2. For now, with the initial english-only
wiki data, we'll be using the pretrained ColBERTv2 weights from Omar's repo.
```
git pull --recurse-submodules
conda env create -f third_party/ColBERT/conda_env.yml
conda activate colbert-v0.4
python index/index_wiki.py
```

## Part 2 - Translate
Translate queries into english.
```
python translate/translate_queries.py
```

## Part 3 - Retrieval
Using the english translated queries, retrieve the top passages from the indexed
english wiki data. Retrieval is done with ColBERT.
```
conda activate colbert-v0.4
python retrieve/retrieve_passages.py
```

## Part 4 - Answer Generation
Using the inital (non-translated) queries and the retrieved english passages,
generate answers with mT5
```
conda activate base

python generate/convert_dpr_retrieval_results_to_seq2seq.py --dev_fp data/colbert_outputs.json --output_dir data/converted_retriever_results

python generate/eval_mgen.py --model_name_or_path data/mgen_mia_train_data_non_iterative_augmented/best_ckpt/ --evaluation_set data/converted_retriever_results/val.source --gold_data_path data/converted_retriever_results/gold_para_qa_data_dev.jsonl --predictions_path data/results/eval_results.txt --gold_data_mode qa --model_type mt5 --max_length 20 --eval_batch_size 4
```

## Part 5 - Evaluation
With the generated answers, evaluate test metrics