import json
import os
import random
from typing import Optional
from optparse import OptionParser

from utils.misc_utils import extract_model_name_from_path

RF = "/home/rfaulk/projects/aip-rgrosse/rfaulk/SCDisc_ICL/data"

DATASET_CONFIG = {
    "semeval_en": {
        "period_1": "corpus1",
        "period_2": "corpus2",
        "threshold": 0.28,
        "period_1_label": "Period 1 (1810-1860)",
        "period_2_label": "Period 2 (1960-2010)",
    },
    "LiverpoolFC": {
        "period_1": "period_2011-13",
        "period_2": "period_2017",
        "threshold": 0.49,
        "period_1_label": "Period 1 (2011-2013)",
        "period_2_label": "Period 2 (2017)",
    },
}

SYSTEM_INSTRUCTION = (
    "You are an expert linguist specializing in lexical semantic change. "
    "Given example usages of a word from two different time periods, "
    "determine whether the word's meaning has changed between the periods. "
    "Answer only Yes or No.")


def format_word_block(
    word: str,
    period_1_sents: list[str],
    period_2_sents: list[str],
    cfg: dict,
    label: Optional[str] = None,
) -> str:
    lines = [f'Word: "{word}"']

    lines.append(f"\nUsages in {cfg['period_1_label']}:")
    for i, s in enumerate(period_1_sents, 1):
        lines.append(f"  {i}. {s}")

    lines.append(f"\nUsages in {cfg['period_2_label']}:")
    for i, s in enumerate(period_2_sents, 1):
        lines.append(f"  {i}. {s}")

    if label is not None:
        lines.append(f"\nDid the meaning change? {label}")
    else:
        lines.append("\nDid the meaning change?")

    return "\n".join(lines)


def build_prompt(
    test_word: str,
    test_contexts: dict,
    icl_examples: list[dict],
    cfg: dict,
) -> str:
    parts = [SYSTEM_INSTRUCTION, ""]

    if icl_examples:
        parts.append("Here are some labeled examples:\n")
        for ex in icl_examples:
            p1_sents = ex["contexts"].get(cfg["period_1"], [])
            p2_sents = ex["contexts"].get(cfg["period_2"], [])
            block = format_word_block(ex["word"],
                                      p1_sents,
                                      p2_sents,
                                      cfg,
                                      label=ex["label"])
            parts.append(block)
            parts.append("")

        parts.append("---")
        parts.append("Now answer for the following word:\n")

    p1 = test_contexts.get(cfg["period_1"], [])
    p2 = test_contexts.get(cfg["period_2"], [])
    parts.append(format_word_block(test_word, p1, p2, cfg, label=None))

    return "\n".join(parts)


def sample_icl_bucket(
    annotated_words: dict,
    contexts: dict,
    n_examples: int,
    threshold: float,
    seed: int,
    cfg: dict,
) -> list[dict]:
    rng = random.Random(seed)

    changed = [
        w for w, s in annotated_words.items() if s > threshold
        and cfg["period_1"] in contexts.get(w.split("_")[0], {})
    ]
    unchanged = [
        w for w, s in annotated_words.items() if s <= threshold
        and cfg["period_1"] in contexts.get(w.split("_")[0], {})
    ]

    n_pos = n_examples // 2
    n_neg = n_examples - n_pos

    rng.shuffle(changed)
    rng.shuffle(unchanged)

    selected_pos = changed[:min(n_pos, len(changed))]
    selected_neg = unchanged[:min(n_neg, len(unchanged))]

    examples = []
    for w in selected_pos:
        clean_w = w.split("_")[0]
        examples.append({
            "word": clean_w,
            "contexts": contexts.get(clean_w, {}),
            "label": "Yes"
        })
    for w in selected_neg:
        clean_w = w.split("_")[0]
        examples.append({
            "word": clean_w,
            "contexts": contexts.get(clean_w, {}),
            "label": "No"
        })

    rng.shuffle(examples)
    return examples


def main():
    parser = OptionParser()
    parser.add_option("--dataset", type=str, default="semeval_en")
    parser.add_option("--tokenizer-model",
                      type=str,
                      default="bert-base-uncased")
    parser.add_option("--max-sents-per-period", type=int, default=5)
    parser.add_option("--context-seed", type=int, default=42)
    parser.add_option("--n-icl-examples", type=int, default=5)
    parser.add_option("--bucket-seed", type=int, default=0)
    options, _ = parser.parse_args()

    dataset = options.dataset
    model_name = extract_model_name_from_path(options.tokenizer_model)
    cfg = DATASET_CONFIG[dataset]

    ctx_file = (
        f"{RF}/{dataset}/icl/contexts__{model_name}"
        f"__n{options.max_sents_per_period}__seed{options.context_seed}.json")
    with open(ctx_file) as f:
        all_contexts = json.load(f)

    with open(f"{RF}/{dataset}/targets.json") as f:
        targets = json.load(f)

    icl_examples = sample_icl_bucket(
        targets,
        all_contexts,
        options.n_icl_examples,
        cfg["threshold"],
        options.bucket_seed,
        cfg,
    )

    icl_words = {ex["word"] for ex in icl_examples}
    prompts = {}
    for word, ctxs in all_contexts.items():
        if word in icl_words:
            continue
        prompt = build_prompt(word, ctxs, icl_examples, cfg)
        prompts[word] = prompt

    outdir = f"{RF}/{dataset}/icl"
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(
        outdir,
        f"prompts__{model_name}__n{options.max_sents_per_period}"
        f"__icl{options.n_icl_examples}__bseed{options.bucket_seed}.json",
    )
    with open(outfile, "w") as f:
        json.dump(prompts, f, indent=2)

    print(
        f"Built {len(prompts)} prompts ({len(icl_examples)} ICL examples each)"
    )
    print(f"Saved to {outfile}")

    sample_word = list(prompts.keys())[0]
    print(f"\n{'='*60}")
    print(f"Sample prompt for '{sample_word}':")
    print(f"{'='*60}")
    print(prompts[sample_word][:2000])


if __name__ == "__main__":
    main()

