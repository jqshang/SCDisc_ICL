# Words for annotations
For annotations, we consider top-15 terms from the rankings induced by select semantic-change-quantification methods. Based on preliminary experiments, we consider rankings from `bert-base-uncased` model and the following methods: `Emb (PRT)`, `Emb (APD)` (for SemEval-EN only), `Subst (JSD)`, `Clustr (AP)` -- see our paper for more details about these methods. For each of these, we also run consider frequency scaling (`+FS`) and part-of-speech matching (`+FS+PM`). For Embeddings-based methods, we also look at the permutation tests (`+PT` and `+PT+FDR`). 

After filtering out of words that **don't** primarily appear as NOUN, VERB, or ADJ part-of-speech, we also run manual checks of the rankings to catch any proper nouns (as well as non-words, offensive language and tokenization errors) that didn't get filtered out in the previous step. These filtered-out words can be found under `word_exclusions.py` (with reasoning in the comments).

The remaining top-15 words by method can be found in `semeval_en__top15_for_annotations.txt`
 and `liverpoolfc__top15_for_annotations.txt`.

# Annotations
Altogether, from these top-15-rankings we have 76 and 83 unique words of SemEval-EN and LiverpoolFC, respectively. For LiverpoolFC, two out of 83 are actually among $T^*$ -- highest-known changes of the target terms set $T$. Annotations for the words from two datasets are in `semeval_en__annotations__main.json` and `liverpoolfc__annotations__main.json`.

For SemEval-EN, eight people contributed annotations. With the exception of `"austerity"` (inadvertently missed by one of the annotators), all words have at least three annotations. Similarly, ten people annotated words of LiverpoolFC, and each word has at least three annotations.

# Additional

In our evaluation, we use threshold $\theta$ to binarize $\ell$($w$), the semantic-change-score averaged accross annotators. The value of $\theta$ is set based on annotations of five high-change and five low/no-change terms from the target set, $T$. These annotations can be found in 
`semeval_en__annotations__targets.json` and `liverpoolfc__annotations__targets.json`. As a result, in SemEval-EN we have $\theta=0.28$ (gives accuracy=0.8 with one false-positive and one false-negative). Similarly, in LiverpoolFC $\theta=0.49$ (gives accuracy=0.9 with one false-negative).   

Finally, we are also releasing data from earlier rounds of annotations. These words have previously been ranked in the top-15, but in the subsequent iterations of the evaluation setup no longer appeared there. We have annotations for 41 and 30 such words from SemEval-EN and LiverpoolFC, respectively: `semeval_en__annotations__extra.json` & `liverpoolfc__annotations__extra.json`.
