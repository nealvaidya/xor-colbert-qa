import jsonlines
import transformers
from tqdm import tqdm
import csv


def read_xor_data(path):
    xor_data = []
    with jsonlines.open(path) as reader:
        for obj in reader:
            xor_data.append(obj)

    xor_data_by_lang = {}
    for query in xor_data:
        lang = query["lang"]
        if lang not in xor_data_by_lang:
            xor_data_by_lang[lang] = []
        xor_data_by_lang[lang].append(query)

    return xor_data_by_lang


def read_mkqa_data(path, languages):
    mkqa_data_by_lang = {}
    for lang in languages:
        queries = []
        with jsonlines.open(f"{path}/mkqa-{lang}.jsonl") as reader:
            for obj in reader:
                queries.append(obj)
        mkqa_data_by_lang[lang] = queries

    return mkqa_data_by_lang


def translate(src_lang, tgt_lang, queries):
    model_name = "Helsinki-NLP/opus-mt-{}-{}".format(src_lang, tgt_lang)
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name, return_tensors="tf"
        )
    except:
        raise RuntimeError(
            "language direction {}-{} is not supported by Hugging Face".format(
                src_lang, tgt_lang
            )
        )
    model = transformers.AutoModelWithLMHead.from_pretrained(model_name)
    model.cuda()
    for query in tqdm(queries):
        data = tokenizer.prepare_seq2seq_batch(query["question"], return_tensors="pt")
        data.to("cuda")
        output = model.generate(**data)
        output = [tokenizer.decode(t, skip_special_tokens=True) for t in output]
        query["translated"] = output


def translate_all(data_by_language):
    for lang in data_by_language:
        if lang == "en":
            for query in tqdm(data_by_language[lang]):
                query["translated"] = query["question"]
        elif lang == "zh_cn":
            translate("zh", "en", data_by_language[lang])
        elif lang == "km":
            translate("mkh", "en", data_by_language[lang])
        elif lang == "te":
            continue
        else:
            translate(lang, "en", data_by_language[lang])
    return


def write_translations(translated_data_by_language, path):
    with open(path, "w") as tsvfile:
        writer = csv.writer(tsvfile, delimiter="\t")
        for lang in translated_data_by_language:
            if lang == "te":
                continue
            for query in translated_data_by_language[lang]:
                writer.writerow(
                    [
                        query["id"],
                        query["translated"][0],
                        query["question"].replace("\t", ""),
                        query["lang"],
                        query["answers"],
                    ]
                )


def main():
    # xor_dev_by_lang = read_xor_data("../data/eval/mia_2022_dev_xorqa.jsonl")
    xor_dev_by_lang = read_xor_data("../../final/data/eval/mia_2022_dev_xorqa.jsonl")
    translate_all(xor_dev_by_lang)
    write_translations(xor_dev_by_lang, "../data/queries_translated.tsv")

    mkqa_langs = ["ar", "en", "es", "fi", "ja", "km", "ko", "ru", "sv", "tr", "zh_cn"]
    # mkqa_dev_by_lang = read_mkqa_data("../data/eval/mkqa_dev", languages=mkqa_langs)
    mkqa_dev_by_lang = read_mkqa_data("../../final/data/eval/mkqa_dev", languages=mkqa_langs)
    translate_all(mkqa_dev_by_lang)
    write_translations(mkqa_dev_by_lang, "../data/mkqa_queries_translated.tsv")


if __name__ == "__main__":
    main()
