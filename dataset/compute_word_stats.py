import os
import re
import json
from optparse import OptionParser
from collections import Counter, defaultdict
from tqdm import tqdm
from misc_utils import *


def main():
    usage = "%prog "
    parser = OptionParser(usage=usage)
    parser.add_option('--dataset', 
                      type=str,
                      default='LiverpoolFC',
                      help='Dataset directory name: default=%default')
    parser.add_option('--tokenizer-model',
                      type=str,
                      default='bert-base-uncased',
                      help='Tokenizer model to use: default=%default')
    parser.add_option('--pre-lemmas-file', 
                      default='', 
                      help='Whether to use lemmas for statistics: default=%default')
      
    (options, args) = parser.parse_args()

    dataset = options.dataset
    tokenizer_model = extract_model_name_from_path(options.tokenizer_model)
    pre_lemmas_file = options.pre_lemmas_file

    indir = '../data/{}/processed_{}'.format(dataset, tokenizer_model)
    outdir = os.path.join(indir, 'stats')
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    print('Loading data...')
    with open(os.path.join(indir, 'tokenized_all.jsonlist')) as f:
        tokenized_data = [json.loads(line) for line in f.readlines()]
    with open(os.path.join(indir, 'lemmatized_all.jsonlist')) as f:
        lemmatized_data = [json.loads(line) for line in f.readlines()]
    with open(os.path.join(indir, 'pos_tagged_all.jsonlist')) as f:
        pos_tagged_data = [json.loads(line) for line in f.readlines()]
    with open(os.path.join(indir, 'lemmas_matching.jsonlist')) as f:
        matching_data = [json.loads(line) for line in f.readlines()]

    assert len(tokenized_data) == len(lemmatized_data) == len(pos_tagged_data) == len(matching_data)

    token_by_pos_counts = {}
    token_by_period_counts = {}
    token_by_period_line_ids = {}
    token_by_pos_by_period_line_ids = {}
    token_by_pos_by_period_sents = {}
    lemma_by_pos_counts = {}
    lemma_by_period_counts = {}
    lemma_pos_by_token_by_period_line_ids = {}

    for line_i in tqdm(range(len(tokenized_data))):
        tokens_dct = tokenized_data[line_i]
        lemmas_dct = lemmatized_data[line_i]
        pos_tags_dct = pos_tagged_data[line_i]
        alignment_dct = matching_data[line_i]
        assert tokens_dct['id'] == lemmas_dct['id'] == pos_tags_dct['id'] == alignment_dct['id']
        source = tokens_dct['source']
        combined_tokens = [''.join(lst) for lst in tokens_dct['tokens']]

        token_id_to_lemma_id_mapping = {}
        for token_id, lemma_id in alignment_dct['alignment']:
            if token_id >= 0 and lemma_id >= 0:
                token_id_to_lemma_id_mapping[token_id] = lemma_id

        string_so_far =''
        for token_id in range(len(combined_tokens)):
            token = combined_tokens[token_id]
            if token_id in token_id_to_lemma_id_mapping:
                if token_id_to_lemma_id_mapping[token_id] >= len(lemmas_dct['lemmas']):
                    print(len(lemmas_dct['lemmas']), len(combined_tokens))
                    print(lemmas_dct['lemmas'])
                    print(combined_tokens)
                    print(alignment_dct['alignment'])
                    print(token_id, token_id_to_lemma_id_mapping[token_id])
                lemma = lemmas_dct['lemmas'][token_id_to_lemma_id_mapping[token_id]]
                pos_tag = pos_tags_dct['pos_tags'][token_id_to_lemma_id_mapping[token_id]]

                if token not in token_by_pos_counts:
                    token_by_pos_counts[token] = {}
                    token_by_period_counts[token] = {}
                    token_by_period_line_ids[token] = {}
                    token_by_pos_by_period_line_ids[token] = {}
                    token_by_pos_by_period_sents[token] = {}
                if pos_tag not in token_by_pos_counts[token]:
                    token_by_pos_counts[token][pos_tag] = 0
                    token_by_pos_by_period_line_ids[token][pos_tag] = {}
                    token_by_pos_by_period_sents[token][pos_tag] = {}
                if source not in token_by_period_counts[token]:
                    token_by_period_counts[token][source] = 0
                    token_by_period_line_ids[token][source] = []
                if source not in token_by_pos_by_period_line_ids[token][pos_tag]:
                    token_by_pos_by_period_line_ids[token][pos_tag][source] = []
                    token_by_pos_by_period_sents[token][pos_tag][source] = []
                token_by_pos_counts[token][pos_tag]  += 1
                token_by_period_counts[token][source] += 1
                token_by_period_line_ids[token][source].append((tokens_dct['id'], token_id))
                token_by_pos_by_period_line_ids[token][pos_tag][source].append((tokens_dct['id'], token_id))
                token_by_pos_by_period_sents[token][pos_tag][source].append((tokens_dct['id'], 
                    '{}:{}'.format(len(string_so_far), len(string_so_far)+len(token)+1)))
                
                if lemma not in lemma_by_pos_counts:
                    lemma_by_pos_counts[lemma] = {}
                    lemma_by_period_counts[lemma] = {}
                if pos_tag not in lemma_by_pos_counts[lemma]:
                    lemma_by_pos_counts[lemma][pos_tag] = 0
                if source not in lemma_by_period_counts[lemma]:
                    lemma_by_period_counts[lemma][source] = 0
                lemma_by_pos_counts[lemma][pos_tag]  += 1
                lemma_by_period_counts[lemma][source] += 1

                lemma_pos = '{}_{}'.format(lemma, pos_tag)
                if lemma_pos not in lemma_pos_by_token_by_period_line_ids:
                    lemma_pos_by_token_by_period_line_ids[lemma_pos] = {}
                if token not in lemma_pos_by_token_by_period_line_ids[lemma_pos]:
                    lemma_pos_by_token_by_period_line_ids[lemma_pos][token] = {}
                if source not in lemma_pos_by_token_by_period_line_ids[lemma_pos][token]:
                    lemma_pos_by_token_by_period_line_ids[lemma_pos][token][source] = []
                lemma_pos_by_token_by_period_line_ids[lemma_pos][token][source].append((tokens_dct['id'], token_id))

            string_so_far = string_so_far + ' ' + token

    print('Writing results into', outdir)
    with open(os.path.join(outdir, 'token_by_pos_frequencies.json'), 'w') as f:
        json.dump(token_by_pos_counts, f)
    # with open(os.path.join(outdir, 'lemma_by_pos_frequencies.json'), 'w') as f:
    #     json.dump(lemma_by_pos_counts, f)
    with open(os.path.join(outdir, 'token_by_source_frequencies.json'), 'w') as f:
        json.dump(token_by_period_counts, f)
    # with open(os.path.join(outdir, 'lemma_by_source_frequencies.json'), 'w') as f:
    #     json.dump(lemma_by_period_counts, f)
    # with open(os.path.join(outdir, 'token_by_source_line_ids.json'), 'w') as f:
    #     json.dump(token_by_period_line_ids, f)
    # with open(os.path.join(outdir, 'token_by_pos_by_source_line_ids.json'), 'w') as f:
    #     json.dump(token_by_pos_by_period_line_ids, f)
    with open(os.path.join(outdir, 'token_by_pos_by_source_sents.json'), 'w') as f:
        json.dump(token_by_pos_by_period_sents, f)
    with open(os.path.join(outdir, 'lemma_pos_by_token_by_source_line_ids.json'), 'w') as f:
        json.dump(lemma_pos_by_token_by_period_line_ids, f)
    

    if pre_lemmas_file != "":
        with open(pre_lemmas_file) as f:
            pre_lemmatized_data = [json.loads(line) for line in f.readlines()]
        with open(os.path.join(indir, 'pre_lemmas_matching.jsonlist')) as f:
            pre_lemma_matching_data = [json.loads(line) for line in f.readlines()]

        assert len(tokenized_data) == len(pre_lemma_matching_data)

        lemma_by_token_by_period_line_ids = {}
        for line_i in tqdm(range(len(tokenized_data))):
            tokens_dct = tokenized_data[line_i]
            lemmas_dct = pre_lemmatized_data[line_i]
            alignment_dct = pre_lemma_matching_data[line_i]
            assert tokens_dct['id'] == lemmas_dct['id'] == alignment_dct['id']
            source = tokens_dct['source']
            combined_tokens = [''.join(lst) for lst in tokens_dct['tokens']]

            token_id_to_lemma_id_mapping = {}
            for token_id, lemma_id in alignment_dct['alignment']:
                if token_id >= 0 and lemma_id >= 0:
                    token_id_to_lemma_id_mapping[token_id] = lemma_id

            for token_id in range(len(combined_tokens)):
                token = combined_tokens[token_id]
                if token_id in token_id_to_lemma_id_mapping:
                    lemma = lemmas_dct['lemmas'][token_id_to_lemma_id_mapping[token_id]]

            if lemma not in lemma_by_token_by_period_line_ids:
                lemma_by_token_by_period_line_ids[lemma] = {}
            if token not in lemma_by_token_by_period_line_ids[lemma]:
                lemma_by_token_by_period_line_ids[lemma][token] = {}
            if source not in lemma_by_token_by_period_line_ids[lemma][token]:
                lemma_by_token_by_period_line_ids[lemma][token][source] = []
            lemma_by_token_by_period_line_ids[lemma][token][source].append((tokens_dct['id'], token_id))

        with open(os.path.join(outdir, 'lemma_by_token_by_source_line_ids.json'), 'w') as f:
            json.dump(lemma_by_token_by_period_line_ids, f)

if __name__ == '__main__':
    main()
