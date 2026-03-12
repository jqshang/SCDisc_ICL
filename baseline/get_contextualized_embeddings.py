import os
import re
import json
import pickle

from transformers import AutoModel, AutoConfig, AutoTokenizer 
import torch
import numpy as np

from optparse import OptionParser
from tqdm import tqdm
from misc_utils import *



# Index select terms in the corpus

def make_batch(tokens_list) :
    mx = max([ len(t) for t in tokens_list ])
    tokens = np.array([ t + ([0] * (mx-len(t))) for t in tokens_list ])
    mask = np.where(tokens != 0, 1, 0)
    return torch.LongTensor(tokens), torch.LongTensor(mask)

def process_batch(token_ids_batch, term_positions_batch):
    token_tensors, segment_tensors = make_batch(token_ids_batch)
    token_tensors = token_tensors.to("cuda")
    segment_tensors = segment_tensors.to("cuda")

    with torch.no_grad():
        outputs = model(token_tensors, segment_tensors, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    hidden_states = torch.stack(hidden_states, dim=0)
    hidden_states = hidden_states.permute(1, 2, 0, 3)
    # print(hidden_states.shape, token_tensors.shape)
    
    embeddings_batch = []
    for sentence_embedding, term_position_info in zip(hidden_states, term_positions_batch):
        # information about token's position as well as the offset due to word-pieces
        position_index, offset = term_position_info
        
        # sum the embeddings of all word pieces
        word_piece_sum_embedding = torch.sum(sentence_embedding[position_index][-4:], dim=0)
        for wp_i in range(1, offset):
            word_piece_sum_embedding += torch.sum(sentence_embedding[position_index+wp_i][-4:], dim=0)
        
        # average embeddings of all wordpieces
        averaged_embedding = word_piece_sum_embedding/offset
        averaged_embedding = averaged_embedding.detach().cpu().numpy()

        embeddings_batch.append(averaged_embedding)

    return embeddings_batch


def main():
    usage = "%prog "
    parser = OptionParser(usage=usage)
    parser.add_option('--dataset', 
                      type=str,
                      default='LiverpoolFC',
                      help='Dataset name: default=%default')
    parser.add_option('--model',
                      type=str,
                      default='bert-base-uncased',
                      help='Model to obtain representations from: default=%default')
    parser.add_option('--term-indices-fname',
                      type=str,
                      default='target_indices.json',
                      help='Name of the file containing terms of interest indexed for the given dataset: default=%default')
    parser.add_option('--terms-start-id',
                      type=int,
                      default=0,
                      help='First id of terms to process: default=%default')
    parser.add_option('--terms-end-id',
                      type=int,
                      default=-1,
                      help='Last id of terms (inclusive) to process: default=%default')
    parser.add_option('--outdir',
                      type=str,
                      default='target_embeddings',
                      help='Directory name for storing extracted contextualized term representations: default=%default')
    parser.add_option('--batch-size',
                      type=int,
                      default=8,
                      help='The batch size for representation extraction: default=%default')
    
    
    (options, args) = parser.parse_args()

    dataset = options.dataset
    model_name = options.model
    term_indices_fname = options.term_indices_fname
    terms_start_id = options.terms_start_id
    terms_end_id = options.terms_end_id
    outdir = options.outdir
    batch_size = options.batch_size

    # Load pretrained model and tokenizer
    model_path = '../models/{}__finetuned__{}'.format(dataset.lower(), extract_model_name_from_path(model_name))
    if not os.path.exists(model_path):
        print('No fine-tuned model under "{}" exists. Attempting to load pre-trained "{}".'.format(model_path, model_name))
        model_path = model_name

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_path, output_hidden_states=True)
    global model
    model = AutoModel.from_pretrained(model_path, config=config)

    model.eval()
    model.to('cuda') # GPU
    print("Model loaded:", model_path)

    # Load tokenized data
    datadir = '../data/{}/processed_{}'.format(dataset, extract_model_name_from_path(model_name))
    print("Loading data...")
    with open(os.path.join(datadir, 'tokenized_all.jsonlist')) as f:
        tokenized_data = [json.loads(line) for line in f.readlines()]
        tokenized_data_dct = {dct['id']: dct for dct in tokenized_data}

    # Load indexed terms for extraction
    with open(os.path.join(datadir, term_indices_fname)) as f:
        terms_indices = json.load(f)

    # Separate a subset of terms to get extract embeddings for
    terms_keys = list(terms_indices.keys())
    select_terms = terms_keys[terms_start_id:terms_end_id] + [terms_keys[terms_end_id]]

    # Prepare outputs directory
    embeddings_dir = '../representations/{}__{}/{}'.format(dataset.lower(), extract_model_name_from_path(model_name), outdir)
    if not os.path.exists(embeddings_dir):
        os.makedirs(embeddings_dir)
        
    
    # Go over words among the list of terms and extract their contextualized embeddings
    unique_tokens = set()
    for term in tqdm(select_terms, mininterval=30):
        if os.path.isfile(os.path.join(embeddings_dir, '{}_embeddings.pickle'.format(term))):
            continue
        indices = terms_indices[term]
        all_embeddings = {}
        token_ids_batch = []
        term_positions_batch = []
        line_ids_batch = []
        for line_id, _, token_position_index, offset in indices:
            processed_tokens = []
            for token_lst in tokenized_data_dct[line_id]['tokens']:
                token = ''.join(token_lst)
                if token != tokenizer.unk_token:
                    processed_tokens += convert_token_lst_to_tokenizer_format(token_lst, model_name)

            processed_tokens = [tokenizer.cls_token] + processed_tokens + [tokenizer.sep_token]
            if len(processed_tokens) > 512:
                continue
            unique_tokens.add('##'.join(processed_tokens[token_position_index+1: token_position_index+1+offset]))
            token_ids = tokenizer.convert_tokens_to_ids(processed_tokens)
            token_ids_batch.append(token_ids)
            term_positions_batch.append((token_position_index+1, offset))
            line_ids_batch.append(line_id)
            if len(token_ids_batch) < batch_size:
                continue
            
            # Once the batch is of batch_size, process the batch
            embeddings_batch = process_batch(token_ids_batch, term_positions_batch)
            assert len(embeddings_batch) == len(line_ids_batch)
            for i in range(len(embeddings_batch)):
                if line_ids_batch[i] not in all_embeddings:
                    all_embeddings[line_ids_batch[i]] = []
                all_embeddings[line_ids_batch[i]].append(embeddings_batch[i])
            
            # refresh and start accumulating the next batch
            token_ids_batch = []
            term_positions_batch = []
            line_ids_batch = []

        # process the last batch
        if len(token_ids_batch) > 0:
            embeddings_batch = process_batch(token_ids_batch, term_positions_batch)
            assert len(embeddings_batch) == len(line_ids_batch)
            for i in range(len(embeddings_batch)):
                if line_ids_batch[i] not in all_embeddings:
                    all_embeddings[line_ids_batch[i]] = []
                all_embeddings[line_ids_batch[i]].append(embeddings_batch[i])

        # collapse embeddings that come from the same line/sentence
        term_embeddings = {}
        for line_id in all_embeddings:
            if len(all_embeddings[line_id]) > 1:
                term_embeddings[line_id] = np.average(all_embeddings[line_id], axis=0)
            elif len(all_embeddings[line_id]) == 1:
                term_embeddings[line_id] = all_embeddings[line_id][0]

        # save term embeddings into a pickle file
        with open(os.path.join(embeddings_dir, '{}_embeddings.pickle'.format(term)), 'wb') as f:
            pickle.dump(term_embeddings, f)

    print("Embeddings extracted for the following tokens: ", sorted(unique_tokens))

            
if __name__ == '__main__':
    main()