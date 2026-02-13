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

