# xor-colbert-qa

## Part 1 - Index Data
Index wikipedia data using ColBERT v2. For now, with the initial english-only
wiki data, we'll be using the pretrained ColBERTv2 weights from Omar's repo.

## Part 2 - Translate
Translate queries into english.

## Part 3 - Retrieval
Using the english translated queries, retrieve the top passages from the indexed
english wiki data. Retrieval is done with ColBERT.

## Part 4 - Answer Generation
Using the inital (non-translated) queries and the retrieved english passages,
generate answers with mT5

## Part 5 - Evaluation
With the generated answers, evaluate test metrics