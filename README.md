# CSC2611 Course Project: Semantic Change Discovery Using In-Context Learning

> Abstract: This project investigates semantic discovery through In-Context Learning (ICL), addressing the critical gap where traditional semantic change detection is often limited to curated target-word lists rather than open-vocabulary discovery in large, real-world corpora. To solve the "needle in a haystack" problem of identifying word-sense shifts in the wild, we propose a methodology that leverages the contextual capabilities of Large Language Models (LLMs), including GPT-4o, Qwen-4B-Thinking, and Llama3, to rank the likelihood of change by prompting them with usage examples from disparate time periods. Evaluating our approach on the SemEval and LiverpoolFC datasets, we found that "thinking models" exhibit a significant scaling effect, with recall reaching approximately 90\% as the number of in-context examples increases. However, our results also reveal a point of diminishing returns where precision tends to degrade with higher example counts, likely due to noise in the prompt examples, underscoring both the high potential of LLM-based ICL for automated linguistic discovery and the necessity for refined example selection strategies.

# Step 1: Data Download

Downloaded and preprocessed datasets can be found at this shared [Google Drive folder](https://drive.google.com/drive/folders/1Qwp5Wt_-5f9pdVf9ifNunXKAYgVOjENK?usp=sharing). You can also run your downloading and preprocessing pipelines as follows.

### SemEval-EN

`mkdir -p data/semeval_en/raw/`

Download and move [SemEval-EN data](https://www.ims.uni-stuttgart.de/en/research/resources/corpora/sem-eval-ulscd-eng/) to `.data/semeval_en/raw/`. Unzip each `corpus1/<lemma or token>/ccoha1.txt.gz` and `corpus2/<lemma or token>/ccoha2.txt.gz`.

```
python dataset/prepare_semevalen_dataset.py
```

### LiverpoolFC

`mkdir -p data/LiverpoolFC/raw/`

Download and move [LiverpoolFC data](https://github.com/marcodel13/Short-term-meaning-shift/tree/master/Dataset) to `.data/LiverpoolFC/raw/`. Unzip `LiverpoolFC_13.txt.zip` and `LiverpoolFC_17.txt.zip`.

```
python dataset/prepare_liverpoolfc_dataset.py
```

# Step 2: Data Preprocessing
```
cd ~/SCDisc_ICL

# 1a. Process raw data
python dataset/process_data.py --dataset semeval_en \
    --infile .data/semeval_en/merged/all.jsonlist \
    --tokenizer-model bert-base-uncased \
    --lemmatize --pos-tag

python dataset/process_data.py --dataset LiverpoolFC \
    --infile .data/LiverpoolFC/clean/all.jsonlist \
    --tokenizer-model bert-base-uncased \
    --lemmatize --pos-tag

# 1b. Map tokens to lemmas + compute word stats
python dataset/match_tokens_to_lemmas.py --dataset semeval_en  --tokenizer-model bert-base-uncased
python dataset/compute_word_stats.py     --dataset semeval_en  --tokenizer-model bert-base-uncased

python dataset/match_tokens_to_lemmas.py --dataset LiverpoolFC --tokenizer-model bert-base-uncased
python dataset/compute_word_stats.py     --dataset LiverpoolFC --tokenizer-model bert-base-uncased

# 1c. Sample control terms
python dataset/sample_control_terms.py --dataset semeval_en  --tokenizer-model bert-base-uncased --control-terms-fname 'controls.json'
python dataset/sample_control_terms.py --dataset LiverpoolFC --tokenizer-model bert-base-uncased --control-terms-fname 'controls.json'

# 1d. Index term occurrences
python dataset/index_term_occurrences.py --dataset semeval_en  --tokenizer-model bert-base-uncased --control-terms-fname 'controls.json' --control-outfile 'control_indices.json'
python dataset/index_term_occurrences.py --dataset LiverpoolFC --tokenizer-model bert-base-uncased --control-terms-fname 'controls.json' --control-outfile 'control_indices.json'
```

# Step 3: LLM Reranking
```
# 3a. Extract contexts
python -m icl.extract_contexts --dataset semeval_en --tokenizer-model bert-base-uncased --max-sents-per-period 5
python -m icl.extract_contexts --dataset LiverpoolFC --tokenizer-model bert-base-uncased --max-sents-per-period 5

# 3b. Build prompts (e.g., 10 ICL examples)
python -m icl.build_prompts --dataset semeval_en --tokenizer-model bert-base-uncased --n-icl-examples 10

# 3c. Run reranking (single run)
python -m icl.run_reranking --dataset semeval_en --tokenizer-model bert-base-uncased --llm-model gemma3 --llm-checkpoint google/gemma-3-4b-it --n-icl-examples 10

# Or: scaling curve across bucket sizes
python -m icl.run_reranking --dataset semeval_en --tokenizer-model bert-base-uncased --llm-model gemma3 --scaling-curve --bucket-sizes 0,1,5,10,20,50 --n-bucket-seeds 3
```