import os
import re
import json
from optparse import OptionParser
from transformers import AutoTokenizer
from tqdm import tqdm
from misc_utils import *

# Index select terms in the corpus


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
    parser.add_option('--target-terms-path',
                      type=str,
                      default='',
                      help='Override for the path to a json file with target terms: default=%default')
    parser.add_option('--control-terms-fname',
                      type=str,
                      default='controls.json',
                      help='The name of a json file to record control terms : default=%default')
    parser.add_option('--control-outfile',
                      type=str,
                      default='control_indices.json',
                      help='Output file to store indices: default=%default')
    parser.add_option('--lemmas', 
                      action="store_true", 
                      help='Whether to use lemmas instead of tokens for statistics: default=%default')
    parser.add_option('--pos-tags', 
                      action="store_true", 
                      help='Whether to compute pos_tag dependent counts: default=%default')
    
    (options, args) = parser.parse_args()

    dataset = options.dataset
    tokenizer_model = extract_model_name_from_path(options.tokenizer_model)
    control_terms_fname = options.control_terms_fname
    target_terms_path = options.target_terms_path
    control_outfile = options.control_outfile
    lemmas = options.lemmas
    pos_tags = options.pos_tags

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(options.tokenizer_model)
    
    datadir = '../data/{}/processed_{}'.format(dataset, tokenizer_model)
    print("Loading data...")
    with open(os.path.join(datadir, 'tokenized_all.jsonlist')) as f:
        tokenized_data = [json.loads(line) for line in f.readlines()]

    if target_terms_path != '':
        with open(target_terms_path) as f:
            target_terms = json.load(f)
    else:
        with open(os.path.join('../data', dataset, 'targets.json')) as f:
            target_terms = json.load(f)

    with open(os.path.join(datadir, control_terms_fname)) as f:
        control_terms = json.load(f)

    indexing_by_token = {}
    for line_dct in tqdm(tokenized_data):
        line_id = line_dct['id']
        source = line_dct['source']
        token_index = 0
        for token_lst in line_dct['tokens']:
            offset = len(token_lst)
            clean_token = ''.join(token_lst)
            if clean_token != tokenizer.unk_token:
                if clean_token not in indexing_by_token:
                    indexing_by_token[clean_token] = []
                indexing_by_token[clean_token].append((line_id, source, token_index, offset))
                token_index += offset


    indexing_by_target_term = {}
    for term in target_terms:
        indexing_by_target_term[term] = indexing_by_token[term]
    with open(os.path.join(datadir, 'target_indices.json'), 'w') as f:
        json.dump(indexing_by_target_term, f)
    
    indexing_by_control_term = {}
    for term in control_terms:
        indexing_by_control_term[term] = indexing_by_token[term]
    with open(os.path.join(datadir, control_outfile), 'w') as f:
        json.dump(indexing_by_control_term, f)




if __name__ == '__main__':
    main()
