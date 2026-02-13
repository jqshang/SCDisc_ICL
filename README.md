# CSC2611 Course Project: Semantic Change Discovery through In-Context Learning

## Setup Virtual Environment

Simply run the following:

```bash
git clone https://github.com/jqshang/SCDisc_ICL
cd SCDisc_ICL
sh setup.sh
```

## Setup Hugging Face API Token

To use the Hugging Face API, you need to generate an access token:

1.  **Log in** to your [Hugging Face account](https://huggingface.co/) [3, 4].
2.  Navigate to **Settings** by clicking your profile picture in the top right corner [2, 6].
3.  Click on **Access Tokens** in the left sidebar [6, 7].
4.  Click the **New token** button [6, 7].
5.  Give your token a name (e.g., "my-app") and select the desired role (e.g., `Read` for inference, `Write` if you are creating models) [1, 6].
6.  Click **Generate a token** [1].
7.  **Copy** the generated token immediately, as it will not be shown again [3, 11].

Store this token securely, for example, in a `.env` file as `HUGGINGFACE_TOKEN=hf_...`.

