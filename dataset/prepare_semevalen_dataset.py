import json
import os
import string
from nltk.tokenize import sent_tokenize

from liverpoolfc_cleaning_utils import *


def main():

    data_dir = '../.data/semeval_en/raw/'

    ## STEP 1: Prepare tokens data

    with open(os.path.join(data_dir, 'corpus1', 'token', 'ccoha1.txt')) as f:
        corpus1_lines = [
            sent.strip().lower() for sent in f.read().split('\n')
            if len(sent) > 0
        ]

    with open(os.path.join(data_dir, 'corpus2', 'token', 'ccoha2.txt')) as f:
        corpus2_lines = [
            sent.strip().lower() for sent in f.read().split('\n')
            if len(sent) > 0
        ]

    dcts_list = []
    line_id = 0
    for line in corpus1_lines:
        dct = {
            'id': "ccoha_{:09d}".format(line_id),
            'text': line,
            'source': 'corpus1'
        }
        dcts_list.append(dct)
        line_id += 1

    for line in corpus2_lines:
        dct = {
            'id': "ccoha_{:09d}".format(line_id),
            'text': line,
            'source': 'corpus2'
        }
        dcts_list.append(dct)
        line_id += 1

    if not os.path.exists('../.data/semeval_en/merged/'):
        os.makedirs('../.data/semeval_en/merged/')

    with open('../.data/semeval_en/merged/all.jsonlist', 'w') as f:
        for dct in dcts_list:
            f.write(json.dumps(dct) + '\n')

    ## STEP 2: Prepare lemmas data

    with open(os.path.join(data_dir, 'corpus1', 'lemma', 'ccoha1.txt')) as f:
        corpus1_lines = [
            sent.strip().lower() for sent in f.read().split('\n')
            if len(sent) > 0
        ]

    with open(os.path.join(data_dir, 'corpus2', 'lemma', 'ccoha2.txt')) as f:
        corpus2_lines = [
            sent.strip().lower() for sent in f.read().split('\n')
            if len(sent) > 0
        ]

    dcts_list = []
    line_id = 0
    for line in corpus1_lines:
        dct = {
            'id': "ccoha_{:09d}".format(line_id),
            'text': line,
            'source': 'corpus1'
        }
        dcts_list.append(dct)
        line_id += 1

    for line in corpus2_lines:
        dct = {
            'id': "ccoha_{:09d}".format(line_id),
            'text': line,
            'source': 'corpus2'
        }
        dcts_list.append(dct)
        line_id += 1

    with open('../.data/semeval_en/merged/all.jsonlist') as f:
        tokenized_data = [json.loads(line) for line in f.readlines()]

    lemmatized_sentences = []
    for i in range(len(dcts_list)):
        sentence_dct = dcts_list[i]
        text = tokenized_data[i]['text'].strip()
        text = re.sub('#+', '#', text)
        text = re.sub('_', ' ', text)
        text = re.sub('▁', ' ', text)
        text = ''.join(ch for ch in text if ch not in string.punctuation)
        text = text.strip()
        if len(text) > 0:
            lemmatized_sentences.append({
                'id':
                sentence_dct['id'],
                'source':
                sentence_dct['source'],
                'lemmas':
                sentence_dct['text'].strip().split()
            })

    with open('../.data/semeval_en/merged/pre_lemmatized_all.jsonlist',
              'w') as f:
        for sent in lemmatized_sentences:
            f.write(json.dumps(sent) + '\n')

    ## STEP 3: Process target words
    truth_dict = {}
    with open('../.data/semeval_en/raw/truth/graded.txt', 'r') as f:
        lines = f.read().split('\n')
        for line in lines:
            if len(line) > 0:
                truth_dict[line.split('\t')[0]] = float(line.split('\t')[1])

    ## The list was derived by mapping lemmas to tokens for every target occurrence
    ## See results after the `process_data.py` script, specifically
    ## `lemma_by_token_by_source_line_ids.json` file
    target_lemma_to_token = {
        'attack_nn': ['attack', 'attacks'],
        'bag_nn': ['bag', 'bags'],
        'ball_nn': ['balls', 'ball'],
        'bit_nn': ['bit', 'bits'],
        'chairman_nn': ['chairman', 'chairmen'],
        'circle_vb': ['circled', 'circling', 'circle'],
        'contemplation_nn': ['contemplation', 'contemplations'],
        'donkey_nn': ['donkey', 'donkeys'],
        'edge_nn': ['edge', 'edges', 'edging'],
        'face_nn': ['faces', 'face'],
        'fiction_nn': ['fiction', 'fictions'],
        'gas_nn': ['gases', 'gas'],
        'graft_nn': ['graft', 'grafts'],
        'head_nn': ['head', 'heads'],
        'land_nn': ['land', 'lands', 'landing'],
        'lane_nn': ['lane', 'lanes'],
        'lass_nn': ['lass', 'lasses', 'lassi'],
        'multitude_nn': ['multitudes', 'multitude'],
        'ounce_nn': ['ounce', 'ounces'],
        'part_nn': ['parts', 'part'],
        'pin_vb': ['pinned', 'pinning'],
        'plane_nn': ['plane', 'planes'],
        'player_nn': ['players', 'player'],
        'prop_nn': ['prop', 'props'],
        'quilt_nn': ['quilt', 'quilting', 'quilts'],
        'rag_nn': ['rags', 'rag'],
        'record_nn': ['record', 'records'],
        'relationship_nn': ['relationship', 'relationships'],
        'risk_nn': ['risk', 'risks'],
        'savage_nn': ['savages', 'savage'],
        'stab_nn': ['stab', 'stabs'],
        'stroke_vb': ['strokes', 'stroked', 'stroking'],
        'thump_nn': ['thump', 'thumping', 'thumps'],
        'tip_vb': ['tipped', 'tipping', 'tip'],
        'tree_nn': ['tree', 'trees'],
        'twist_nn': ['twist', 'twists'],
        'word_nn': ['word', 'words']
    }

    targets_to_true_score = {}
    token_to_lemma = {}
    for lemma_pos in target_lemma_to_token:
        for token in target_lemma_to_token[lemma_pos]:
            targets_to_true_score[token] = truth_dict[lemma_pos]
            token_to_lemma[token] = lemma_pos

    with open('../.data/semeval_en/targets.json', 'w') as f:
        json.dump(targets_to_true_score, f)

    with open('../.data/semeval_en/token_to_lemma.json', 'w') as f:
        json.dump(token_to_lemma, f)


if __name__ == '__main__':
    main()
