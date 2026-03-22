import json
import os
import sys
import string
from optparse import OptionParser
from nltk.tokenize import sent_tokenize
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dataset.liverpoolfc_cleaning_utils import *


def main():
    usage = "usage: %prog [options]"
    parser = OptionParser(usage)
    parser.add_option('--data-dir',
                      type=str,
                      default='.data',
                      help='Root data directory: default=%default')
    (options, args) = parser.parse_args()
    data_dir = options.data_dir

    ## STEP 1: Clean up the data [follows main() from liverpoolfc_cleaning_utils.py]

    raw_data_dir = os.path.join(data_dir, 'LiverpoolFC', 'raw') + '/'
    good_chars = set(string.ascii_lowercase + string.digits +
                     string.punctuation + ' ')

    lines_13 = []
    for text in process_LiverpoolFC(
            os.path.join(raw_data_dir, 'LiverpoolFC_13.txt')):
        for sent in sent_tokenize(text):
            sent = "".join([c if c in good_chars else '' for c in sent])
            if len(sent) > 0:
                lines_13.append(sent)

    lines_17 = []
    for text in process_LiverpoolFC(
            os.path.join(raw_data_dir, 'LiverpoolFC_17.txt')):
        for sent in sent_tokenize(text):
            sent = "".join([c if c in good_chars else '' for c in sent])
            if len(sent) > 0:
                lines_17.append(sent)

    ## STEP 2: Record the dataset under the given format

    dcts_list = []
    sent_id = 0
    for sent in lines_13:
        dct = {
            'id': f"lvfc_{sent_id:09d}",
            'text': sent,
            'source': 'period_2011-13'
        }
        dcts_list.append(dct)
        sent_id += 1

    for sent in lines_17:
        dct = {
            'id': f"lvfc_{sent_id:09d}",
            'text': sent,
            'source': 'period_2017'
        }
        dcts_list.append(dct)
        sent_id += 1

    clean_dir = os.path.join(data_dir, 'LiverpoolFC', 'clean')
    if not os.path.exists(clean_dir):
        os.makedirs(clean_dir)

    with open(os.path.join(clean_dir, 'all.jsonlist'), 'w') as f:
        for dct in dcts_list:
            f.write(json.dumps(dct) + '\n')

    ## STEP 3: Process target words
    df = pd.read_csv(os.path.join(data_dir, 'LiverpoolFC', 'liverpool_annotated_words.csv'))[[
        'word', 'shift_index'
    ]]
    temp_dct = df.to_dict('index')
    targets_dct = {
        temp_dct[id_]['word']: temp_dct[id_]['shift_index']
        for id_ in temp_dct
    }

    with open(os.path.join(data_dir, 'LiverpoolFC', 'targets.json'), 'w') as f:
        json.dump(targets_dct, f)


if __name__ == '__main__':
    main()
