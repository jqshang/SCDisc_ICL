# CSC2611 Course Project: Semantic Change Discovery through In-Context Learning

## Setup Virtual Environment

Simply run the following:

```bash
git clone https://github.com/jqshang/SCDisc_ICL
cd SCDisc_ICL
sh setup.sh
```

During setup you may be prompted to enter your HuggingFace API key in order to access models.

## Setup Hugging Face API Token

To use the Hugging Face API, you need to generate an access token:

1.  **Log in** to your [Hugging Face account](https://huggingface.co/).
2.  Navigate to **Settings** by clicking your profile picture in the top right corner.
3.  Click on **Access Tokens** in the left sidebar.
4.  Click the **New token** button.
5.  Give your token a name (e.g., "my-app") and select the desired role (e.g., `Read` for inference, `Write` if you are creating models).
6.  Click **Generate a token**.
7.  **Copy** the generated token immediately, as it will not be shown again.

Store this token securely, for example, in a `.env` file as `HUGGINGFACE_TOKEN=hf_...`.

You can further ensure that HF models are stored consistently by exporting `$HF_HOME`:

```bash
export HF_HOME="/scratch/$USER/hf_cache"
```

## Testing Out a Basic Model

You may run a query on GPT-2 by running:

```bash
sh launch_test.sh
```

If you want to run on an L40 GPU you can use *slurm* workload manager:

```
sbatch launch_test.sh
```

# Step 1: Data Download
### SemEval-EN

`mkdir -p data/semeval_en/raw/`

Download and move [SemEval-EN data](https://www.ims.uni-stuttgart.de/en/research/resources/corpora/sem-eval-ulscd-eng/) to `data/semeval_en/raw/`. Unzip each `corpus1/<lemma or token>/ccoha1.txt.gz` and `corpus2/<lemma or token>/ccoha2.txt.gz`.

```
python dataset/prepare_semevalen_dataset.py
```

### LiverpoolFC

`mkdir -p data/LiverpoolFC/raw/`

Download and move [LiverpoolFC data](https://github.com/marcodel13/Short-term-meaning-shift/tree/master/Dataset) to `data/LiverpoolFC/raw/`. Unzip `LiverpoolFC_13.txt.zip` and `LiverpoolFC_17.txt.zip`.

```
python dataset/prepare_liverpoolfc_dataset.py
```

# Step 2: Data Preprocessing
```
cd ~/SCDisc_ICL

# 1a. Process raw data
python dataset/process_data.py --dataset semeval_en \
    --infile ../.data/semeval_en/merged/all.jsonlist \
    --tokenizer-model bert-base-uncased \
    --lemmatize --pos-tag

python dataset/process_data.py --dataset LiverpoolFC \
    --infile ../.data/LiverpoolFC/clean/all.jsonlist \
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