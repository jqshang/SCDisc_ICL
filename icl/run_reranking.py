import json
from optparse import OptionParser
import os
import time

from icl.build_prompts import (
    DATASET_CONFIG,
    build_prompt,
    sample_icl_bucket,
)
from model import get_model
from tqdm import tqdm
from utils.misc_utils import extract_model_name_from_path


def score_all_words(llm, prompts: dict[str, str]) -> dict[str, float]:
    scores = {}
    for word, prompt in tqdm(prompts.items(), desc="Scoring words"):
        try:
            scores[word] = llm.score_change(prompt)
        except Exception as e:
            print(f"  [WARN] Failed on '{word}': {e}")
            scores[word] = 0.5
    return scores


def rank_words(scores: dict[str, float]) -> list[str]:
    return sorted(scores, key=lambda w: scores[w], reverse=True)


def evaluate_ranking(
    ranked_words: list[str],
    target_pos: set[str],
    top_k_values: list[int] = None,
) -> dict[str, float]:
    """Evaluate the ranking of words."""
    # if top_k_values is None:
    #     top_k_values = [10, 15, 25, 50, 100, 250, 500, 1000]
    results = {}
    for k in top_k_values:
        top_k = set(ranked_words[:k])
        discovered = top_k.intersection(target_pos)
        precision = (len(discovered) /
                     min(k, len(ranked_words)) if ranked_words else 0)
        recall = len(discovered) / len(target_pos) if target_pos else 0
        results[f"top_{k}"] = {
            "discovered": len(discovered),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        }
    rank_map = {w: i for i, w in enumerate(ranked_words)}
    target_pos_ranks = [rank_map.get(w, len(ranked_words)) for w in target_pos]
    results["avg_rank_T_pos"] = (round(
        sum(target_pos_ranks) /
        len(target_pos_ranks), 2) if target_pos_ranks else None)
    return results


def evaluate_discovery(
    scores: dict[str, float],
    target_pos: set[str],
    target_neg: set[str],
) -> dict[str, float]:
    change = {w for w, s in scores.items() if s > 0.5}
    no_change = {w for w, s in scores.items() if s <= 0.5}
    tp = change.intersection(target_pos)  # true positives
    tn = no_change.intersection(target_neg)  # true negatives
    fp = target_neg - no_change
    return {
        "true positives": str(tp),
        "true negatives": str(tn),
        "false positives": str(fp),
        "precision": len(tp) / (len(tp) + len(fp)) if tp else 0.,
        "recall": len(tp) / len(target_pos) if target_pos else 0.,
    }


def main():
    parser = OptionParser()
    parser.add_option("--dataset", type=str, default="semeval_en")
    parser.add_option("--tokenizer-model",
                      type=str,
                      default="bert-base-uncased")
    parser.add_option(
        "--llm-model",
        type=str,
        default="gemma3",
        help="Model key: gemma3, llama3, qwen, gpt4, deepseek-r1",
    )
    parser.add_option(
        "--llm-checkpoint",
        type=str,
        default="",
        help="Override HuggingFace model ID / path",
    )
    parser.add_option("--max-sents-per-period", type=int, default=5)
    parser.add_option("--context-seed", type=int, default=42)
    parser.add_option("--n-icl-examples", type=int, default=5)
    parser.add_option("--bucket-seed", type=int, default=0)
    parser.add_option(
        "--scaling-curve",
        action="store_true",
        default=False,
        help="Run across multiple bucket sizes and seeds",
    )
    parser.add_option(
        "--bucket-sizes",
        type=str,
        default="0,1,5,10,20,50",
        help="Comma-separated bucket sizes for scaling curve",
    )
    parser.add_option("--n-bucket-seeds", type=int, default=3)
    parser.add_option("--data-dir",
                      type=str,
                      default=".data",
                      help="Root data directory: default=%default")
    parser.add_option(
        "--output-dir",
        type=str,
        default=None,
        help=
        "Output directory (default: results/icl_reranking/<dataset>_<llm-model>_<llm-checkpoint>)"
    )
    options, _ = parser.parse_args()

    dataset = options.dataset
    tok_model = extract_model_name_from_path(options.tokenizer_model)
    cfg = DATASET_CONFIG[dataset]
    data_dir = options.data_dir

    ctx_file = os.path.join(
        data_dir, dataset, "icl", f"contexts__{tok_model}"
        f"__n{options.max_sents_per_period}__seed{options.context_seed}.json")
    with open(ctx_file) as f:
        all_contexts = json.load(f)

    with open(os.path.join(data_dir, dataset, "targets.json")) as f:
        targets = json.load(f)

    # Split the targets along their discovery classification.
    target_pos = {
        w.split("_")[0]
        for w, s in targets.items() if s >= cfg["threshold"]
    }
    target_neg = {
        w.split("_")[0]
        for w, s in targets.items() if s < cfg["threshold"]
    }
    target_tot = target_pos.union(target_neg)

    llm_kwargs = {}
    if options.llm_checkpoint:
        llm_kwargs["model_name_or_path"] = options.llm_checkpoint
    llm = get_model(options.llm_model, **llm_kwargs)
    print(f"Loading {llm} ...")
    llm.load()
    print("Model loaded.")

    results_dir = (
        options.output_dir if options.output_dir else
        f"results/icl_reranking/{dataset}_{options.llm_model}_{options.llm_checkpoint}"
    )
    os.makedirs(results_dir, exist_ok=True)

    if options.scaling_curve:
        bucket_sizes = [int(x) for x in options.bucket_sizes.split(",")]
        all_scaling_results = {}

        for n_icl in bucket_sizes:
            all_scaling_results[n_icl] = {}
            for bseed in range(options.n_bucket_seeds):
                print(f"\n{'='*60}")
                print(f"  n_icl={n_icl}  bucket_seed={bseed}")
                print(f"{'='*60}")

                if n_icl == 0:
                    icl_examples = []
                else:
                    icl_examples = sample_icl_bucket(
                        targets,
                        all_contexts,
                        n_icl,
                        cfg["threshold"],
                        bseed,
                        cfg,
                    )
                icl_words = {ex["word"] for ex in icl_examples}
                prompts = {}
                for word, ctxs in all_contexts.items():
                    if word in icl_words or word not in target_tot:
                        continue
                    prompts[word] = build_prompt(word, ctxs, icl_examples, cfg)

                scores = score_all_words(llm, prompts)
                print(scores)
                ranked = rank_words(scores)
                # eval_results = evaluate_ranking(ranked, target_pos, target_neg)
                eval_results = evaluate_discovery(scores, target_pos,
                                                  target_neg)
                all_scaling_results[n_icl][bseed] = {
                    "scores": scores,
                    "ranking": ranked,
                    "evaluation": eval_results,
                }
                print(json.dumps(eval_results, indent=2))

        outfile = os.path.join(results_dir, "scaling_curve_results.json")
        with open(outfile, "w") as f:
            json.dump(all_scaling_results, f, indent=2)
        print(f"\nScaling curve results saved to {outfile}")

    else:
        n_icl = options.n_icl_examples
        bseed = options.bucket_seed

        if n_icl == 0:
            icl_examples = []
        else:
            icl_examples = sample_icl_bucket(
                targets,
                all_contexts,
                n_icl,
                cfg["threshold"],
                bseed,
                cfg,
            )
        icl_words = {ex["word"] for ex in icl_examples}
        prompts = {}
        # TODO: Validation set.
        stored_prompt = None
        for word, ctxs in all_contexts.items():
            # Only include target words.
            if word in icl_words or word not in target_tot:
                continue
            prompts[word] = build_prompt(word, ctxs, icl_examples, cfg)
            stored_prompt = prompts[word]

        scores = score_all_words(llm, prompts)
        print(scores)
        if stored_prompt is not None:
            print(stored_prompt)
        ranked = rank_words(scores)
        # eval_results = evaluate_ranking(ranked, target_pos, target_neg)
        eval_results = evaluate_discovery(scores, target_pos, target_neg)

        scores_file = os.path.join(
            results_dir,
            f"scores__icl{n_icl}__bseed{bseed}.json",
        )
        with open(scores_file, "w") as f:
            json.dump(scores, f, indent=2)

        ranking_file = os.path.join(
            results_dir,
            f"ranking__icl{n_icl}__bseed{bseed}.json",
        )
        with open(ranking_file, "w") as f:
            json.dump(ranked, f, indent=2)

        eval_file = os.path.join(
            results_dir,
            f"evaluation__icl{n_icl}__bseed{bseed}.json",
        )
        with open(eval_file, "w") as f:
            json.dump(eval_results, f, indent=2)

        print(f"\nResults for {options.llm_model} | {dataset} | "
              f"n_icl={n_icl} | seed={bseed}:")
        print(json.dumps(eval_results, indent=2))
        print(f"\nFiles saved to {results_dir}/")


if __name__ == "__main__":
    main()
