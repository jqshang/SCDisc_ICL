import os
import re
import pickle
import json
from optparse import OptionParser

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM
from misc_utils import *



def make_batch(tokens_list) :
	mx = max([ len(t) for t in tokens_list ])
	tokens = np.array([ t + ([0] * (mx-len(t))) for t in tokens_list ])
	mask = np.where(tokens != 0, 1, 0)
	return torch.LongTensor(tokens), torch.LongTensor(mask)


def process_batch(token_ids_batch, top_k, line_ids_batch, all_substitutes):
	token_tensors, segment_tensors = make_batch(token_ids_batch)
	mask_token_index = torch.where(token_tensors == tokenizer.mask_token_id, True, False)
	token_tensors = token_tensors.to("cuda")
	segment_tensors = segment_tensors.to("cuda")
	
	with torch.no_grad():
		token_logits = model(token_tensors, segment_tensors).logits.detach().cpu()
	
	mask_token_logits = token_logits[mask_token_index]
	# extract extra top-k words to account for filtering of stopwords down the line
	topk_tokens = torch.topk(mask_token_logits, top_k+10, dim=1)
	topk_token_indices = topk_tokens.indices.unsqueeze(-1).tolist()
	topk_token_logits = topk_tokens.values.tolist()

	assert len(topk_token_indices) == len(line_ids_batch)
	for row_i in range(len(topk_token_indices)):
		decoded_tokens = tokenizer.batch_decode(topk_token_indices[row_i]) 
		if line_ids_batch[row_i] not in all_substitutes:
			all_substitutes[line_ids_batch[row_i]] = []
		token_logit_tuples = list(zip(decoded_tokens, topk_token_logits[row_i]))
		# filter out stopwords
		token_logit_tuples_filtered = [tup for tup in token_logit_tuples 
									   if tup[1] not in STOPWORDS]
		all_substitutes[line_ids_batch[row_i]] += token_logit_tuples_filtered[:top_k]   

	return all_substitutes



def main():
	"""Script for running Card (2023) extraction of top substitute 
	candidates for masked target token"""
	usage = "%prog"
	parser = OptionParser(usage=usage)
	parser.add_option('--dataset', 
					  type=str, 
					  default='LiverpoolFC',
					  help='Dataset name: default=%default')
	parser.add_option('--model', 
					  type=str, 
					  default='bert-base-uncased',
					  help='Model to obtain mask token substitutes from: default=%default')
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
					  default='target_substitutes',
					  help='Directory name for storing extracted contextualized term representations: default=%default')
	parser.add_option('--batch-size',
					  type=int,
					  default=8,
					  help='The batch size for representation extraction: default=%default')
	parser.add_option('--max-window-size', 
					  type=int, 
					  default=50,
					  help='Max window radius (in word tokens): default=%default')
	parser.add_option('--top-k',
					  type=int, 
					  default=5,
					  help='Top-k terms to keep: default=%default')
	parser.add_option('--stopwords-file-path', 
					  type=str, 
					  default='../data/stopwords/STOPWORDS_spacy_en_core_web_sm.txt',
					  help='Stopwords .txt file: default=%default')


	(options, args) = parser.parse_args()

	dataset = options.dataset
	model_name = options.model
	term_indices_fname = options.term_indices_fname
	terms_start_id = options.terms_start_id
	terms_end_id = options.terms_end_id
	outdir = options.outdir
	batch_size = options.batch_size
	max_window_size = options.max_window_size
	top_k = options.top_k
	stopwords_file = options.stopwords_file_path
	
	# Load stopwords
	global STOPWORDS
	try:
		with open(stopwords_file) as f:
			STOPWORDS = f.read().split('\n')
	except FileNotFoundError: 
		STOPWORDS = []

	# Load pretrained model and tokenizer
	model_path = '../models/{}__finetuned__{}'.format(dataset.lower(), extract_model_name_from_path(model_name))
	if not os.path.exists(model_path):
		print('No fine-tuned model under "{}" exists. Attempting to load pre-trained "{}".'.format(model_path, model_name))
		model_path = model_name

	global tokenizer
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	global model
	model = AutoModelForMaskedLM.from_pretrained(model_path)

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
	terms_keys = sorted(list(terms_indices.keys()))
	select_terms = terms_keys[terms_start_id:terms_end_id] + [terms_keys[terms_end_id]]

	# Prepare outputs directory
	substitutes_dir = '../representations/{}__{}/{}'.format(dataset.lower(), extract_model_name_from_path(model_name), outdir)
	if not os.path.exists(substitutes_dir):
		os.makedirs(substitutes_dir)
		
	
	# Go over words among the list of terms and extract their contextualized embeddings
	unique_tokens = set()
	# substitutes_extracted = [f.split('_')[0] for f in os.listdir(substitutes_dir)]
	for term in tqdm(select_terms, mininterval=30):
		if os.path.isfile(os.path.join(substitutes_dir, '{}_substitutes.pickle'.format(term))):
			continue
		indices = terms_indices[term]
		all_substitutes = {}
		token_ids_batch = []
		term_positions_batch = []
		line_ids_batch = []
		for line_id, _, _, _ in indices:
			processed_tokens_left = []
			processed_tokens_right = []
			seen_masked_term = False
			for token_lst in tokenized_data_dct[line_id]['tokens']:
				token = ''.join(token_lst)
				if token == term:
					seen_masked_term = True
					unique_tokens.add(token)
				# separate word pieces
				elif token != tokenizer.unk_token:
					if seen_masked_term:
						processed_tokens_right += convert_token_lst_to_tokenizer_format(token_lst, model_name)

					else:
						processed_tokens_left += convert_token_lst_to_tokenizer_format(token_lst, model_name)
			
			processed_tokens =  [tokenizer.cls_token] 
			processed_tokens += processed_tokens_left[-max_window_size:]
			processed_tokens += [tokenizer.mask_token]
			processed_tokens += processed_tokens_right[:max_window_size]
			processed_tokens += [tokenizer.sep_token]
			if len(processed_tokens) > 512:
				continue

			token_ids = tokenizer.convert_tokens_to_ids(processed_tokens)
			token_ids_batch.append(token_ids)
			line_ids_batch.append(line_id)
			if len(token_ids_batch) < batch_size:
				continue
			
			# Once the batch is of batch_size, process the batch
			all_substitutes = process_batch(token_ids_batch, top_k, line_ids_batch, all_substitutes)

			# refresh and start accumulating the next batch
			token_ids_batch = []
			term_positions_batch = []
			line_ids_batch = []

		# process the last batch
		if len(token_ids_batch) > 0:
			all_substitutes = process_batch(token_ids_batch, top_k, line_ids_batch, all_substitutes)
			

		# collapse embeddings that come from the same line/sentence and shorten the list to only include to
		term_substitutes = {}
		for line_id in all_substitutes:
			if len(all_substitutes[line_id]) > top_k:
				term_substitutes[line_id] = sorted(all_substitutes[line_id], 
												   key=lambda t: t[1],
												   reverse=True)[:top_k]
			elif len(all_substitutes[line_id]) == top_k:
				term_substitutes[line_id] = all_substitutes[line_id]

		# save term embeddings into a pickle file
		with open(os.path.join(substitutes_dir, '{}_substitutes.pickle'.format(term)), 'wb') as f:
			pickle.dump(term_substitutes, f)

	print("Embeddings extracted for the following tokens: ", sorted(unique_tokens))



if __name__ == '__main__':
	main()

