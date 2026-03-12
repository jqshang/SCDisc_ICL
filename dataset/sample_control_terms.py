import os
import json
import random
from optparse import OptionParser
from collections import Counter, defaultdict
from transformers import AutoTokenizer
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
    parser.add_option('--target-terms-path',
                      type=str,
                      default='',
                      help='Override for the path to a json file with target terms: default=%default')
    parser.add_option('--control-terms-fname',
                      type=str,
                      default='controls.json',
                      help='The name of a json file to record control terms : default=%default')
    parser.add_option('--use-lemmas', 
                      action="store_true", 
                      help='Whether to use lemmas instead of tokens for statistics: default=%default')
    parser.add_option('--use-pos-tags', 
                      action="store_true", 
                      help='Whether to compute pos_tag dependent counts: default=%default')
    parser.add_option('--min-count', type=int, default=20,
                      help='Restrict token selection to those with at least this many tokens: default=%default')
    parser.add_option('--min-count-per-source', type=int, default=3,
                      help='Restrict token selection to those with at least this many tokens per corpus: default=%default')
    
    
    
    (options, args) = parser.parse_args()

    dataset = options.dataset
    tokenizer_model = extract_model_name_from_path(options.tokenizer_model)
    control_terms_fname = options.control_terms_fname
    target_terms_path = options.target_terms_path
    use_lemmas = options.use_lemmas
    use_pos_tags = options.use_pos_tags
    min_count = options.min_count
    min_count_per_source = options.min_count_per_source
    
    if target_terms_path != '':
        with open(target_terms_path) as f:
            targets = json.load(f)
    else:
        with open(os.path.join('../data', dataset, 'targets.json')) as f:
            targets = json.load(f)

    datadir = '../data/{}/processed_{}'.format(dataset, tokenizer_model)
    with open(os.path.join(datadir, 'stats', 'token_by_source_frequencies.json')) as f:
        frequencies_dct = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(options.tokenizer_model)
    
    control_terms_dct = {} 
    for target_term in tqdm(targets):
        if target_term in frequencies_dct:
            target_freq = sum(frequencies_dct[target_term].values())
            for term in frequencies_dct:
                if term not in targets and term != tokenizer.unk_token:
                    term_freq = sum(frequencies_dct[term].values())
                    term_freq_binary = all([frequencies_dct[term][s]>min_count_per_source 
                                            for s in frequencies_dct[term]])\
                                        and term_freq > min_count
                    if term_freq_binary and (target_freq*0.5 <= term_freq <= target_freq*2):
                        control_terms_dct[term] = True
    control_terms = list(control_terms_dct.keys())
    print(type(control_terms))
    random.shuffle(control_terms)

    outfile = os.path.join(datadir, control_terms_fname)
    print(len(control_terms), 'control terms selected and written into', outfile)
    with open(outfile, 'w') as f:
        json.dump(control_terms, f)




if __name__ == '__main__':
    main()
