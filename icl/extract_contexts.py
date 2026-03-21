import json
import os
import random
from optparse import OptionParser

from tqdm import tqdm
from utils.misc_utils import extract_model_name_from_path, combine_tokens_into_clean_text

RF = '/home/rfaulk/projects/aip-rgrosse/rfaulk/SCDisc_ICL/data'


def load_corpus_by_line_id(tokenized_path: str, model_name: str):
    corpus = {}
    with open(tokenized_path) as f:
        for line in tqdm(f, desc="Loading corpus"):
            dct = json.loads(line)
            line_id = dct["id"]
            source = dct["source"]
            flat_tokens = [
                tk for token_lst in dct["tokens"] for tk in token_lst
            ]
            text = combine_tokens_into_clean_text(flat_tokens, model_name)
            corpus[line_id] = (source, text)
    return corpus


def extract_contexts_for_term(
    term: str,
    occurrences: list,
    corpus: dict,
    max_sents: int,
    rng: random.Random,
):
    by_period: dict[str, list[str]] = {}
    for line_id, source, token_index, offset in occurrences:
        if line_id not in corpus:
            continue
        _, text = corpus[line_id]
        if source not in by_period:
            by_period[source] = []
        by_period[source].append(text)

    for period in by_period:
        if len(by_period[period]) > max_sents:
            by_period[period] = rng.sample(by_period[period], max_sents)

    return by_period


def main():
    parser = OptionParser()
    parser.add_option("--dataset", type=str, default="semeval_en")
    parser.add_option("--tokenizer-model",
                      type=str,
                      default="bert-base-uncased")
    parser.add_option(
        "--max-sents-per-period",
        type=int,
        default=5,
        help="Max sentences to sample per time period per word (default 5)",
    )
    parser.add_option("--seed", type=int, default=42)
    parser.add_option(
        "--control-terms-fname",
        type=str,
        default="controls.json",
    )
    options, _ = parser.parse_args()

    dataset = options.dataset
    model_name = extract_model_name_from_path(options.tokenizer_model)
    max_sents = options.max_sents_per_period
    rng = random.Random(options.seed)

    datadir = f"{RF}/{dataset}/processed_{model_name}"
    outdir = f"{RF}/{dataset}/icl"
    os.makedirs(outdir, exist_ok=True)

    corpus = load_corpus_by_line_id(
        os.path.join(datadir, "tokenized_all.jsonlist"), model_name)

    with open(os.path.join(datadir, "target_indices.json")) as f:
        target_indices = json.load(f)

    with open(os.path.join(datadir, "control_indices.json")) as f:
        control_indices = json.load(f)

    all_contexts = {}

    for term, occs in tqdm(target_indices.items(), desc="Targets"):
        all_contexts[term] = extract_contexts_for_term(term, occs, corpus,
                                                       max_sents, rng)

    for term, occs in tqdm(control_indices.items(), desc="Controls"):
        if term not in all_contexts:
            all_contexts[term] = extract_contexts_for_term(
                term, occs, corpus, max_sents, rng)

    outfile = os.path.join(
        outdir,
        f"contexts__{model_name}__n{max_sents}__seed{options.seed}.json")
    with open(outfile, "w") as f:
        json.dump(all_contexts, f, indent=2)

    print(f"Saved {len(all_contexts)} term contexts to {outfile}")


if __name__ == "__main__":
    main()

