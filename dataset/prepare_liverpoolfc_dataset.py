import json
import os
import string
from nltk.tokenize import sent_tokenize
import pandas as pd

from dataset.liverpoolfc_cleaning_utils import *

RF = '/home/rfaulk/projects/aip-rgrosse/rfaulk/SCDisc_ICL/data/LiverpoolFC'


def main():

    ## STEP 1: Clean up the data [follows main() from liverpoolfc_cleaning_utils.py]

    raw_data_dir = f'{RF}/raw/'
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
            'id': "lvfc_{:09d}".format(sent_id),
            'text': sent,
            'source': 'period_2011-13'
        }
        dcts_list.append(dct)
        sent_id += 1

    for sent in lines_17:
        dct = {
            'id': "lvfc_{:09d}".format(sent_id),
            'text': sent,
            'source': 'period_2017'
        }
        dcts_list.append(dct)
        sent_id += 1

    if not os.path.exists(f'{RF}/clean'):
        os.makedirs(f'{RF}/clean')

    with open(f'{RF}/clean/all.jsonlist', 'w') as f:
        for dct in dcts_list:
            f.write(json.dumps(dct) + '\n')

    ## STEP 3: Process target words
    df = pd.read_csv(f'{RF}/liverpool_annotated_words.csv')[[
        'word', 'shift_index'
    ]]
    temp_dct = df.to_dict('index')
    targets_dct = {
        temp_dct[id_]['word']: temp_dct[id_]['shift_index']
        for id_ in temp_dct
    }

    with open(f'{RF}/targets.json', 'w') as f:
        json.dump(targets_dct, f)


if __name__ == '__main__':
    main()

