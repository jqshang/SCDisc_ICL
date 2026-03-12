import re
import os
import json
import spacy
from tqdm import tqdm
from optparse import OptionParser
from transformers import AutoTokenizer
from misc_utils import *
import string


def main():
	"""Scrpt to process the data. By default it tokenizes sentences from the corpora. 
	Optionally, sentences can be lemmatized and tokens can be tagged for part of speech
	"""
	usage = "usage: %prog [options] arg"
	parser = OptionParser(usage)
	parser.add_option('--dataset', 
					  type=str, 
					  default='LiverpoolFC',
					  help='Dataset directory name: default=%default')
	parser.add_option('--infile',
					  type=str,
					  default='../data/LiverpoolFC/clean/all.jsonlist',
					  help='Input file with clean data from all period corpora: default=%default.\
							The data is expected to be stored in a jsonlist format, where each\
							line is a dictionary correponding to an individual sentence. Its text\
							is stored under "text" field, its ID under "id" field, and the period\
							it comes from under "source" filed. For example:\n\
								{"id": "000000001", \
								"text": "my opinion is best summarised here", \
								"source": "period_2011-13"}')
	parser.add_option('--tokenizer-model', 
					  type=str, 
					  default='bert-base-uncased',
					  help='Model name/path to be utilized during tokenization: default=%default')
	parser.add_option('--lemmatize', 
					  action="store_true", 
					  help='Whether or not to lemmatize the data: default=%default')
	parser.add_option('--use-existing-lemmas', 
					  action="store_false", 
					  help='Whether or not to use pre-lemmatized data: default=%default')
	parser.add_option('--pos-tag', 
					  action="store_true", 
					  help='Whether or not to run part-of-speech tagging on the tokens: default=%default')
	parser.add_option('--verbose',
					  action='store_true',
					  help='Whether to print out misaligned tokens and lemmas/pos-tags: default=%default')
	parser.add_option('--spacy_model', 
					  type=str, 
					  default='en_core_web_sm',
					  help='Which spaCy model to use for lemmatization or pos-tagging: default=%default')

	

	(options, args) = parser.parse_args()

	dataset = options.dataset
	infile = options.infile
	model_name = options.tokenizer_model
	lemmatization = options.lemmatize
	use_existing_lemmas = options.use_existing_lemmas
	pos_tagging = options.pos_tag
	verbose = options.verbose
	spacy_model = options.spacy_model

	outdir = os.path.join('../data', dataset, 'processed_' + extract_model_name_from_path(model_name))
	if not os.path.exists(outdir):
		os.makedirs(outdir)

	with open(infile) as f:
		sentences = f.readlines()
	print('Loaded n={} sentences from {} file'.format(len(sentences), infile))
	

	print('Loading the tokenizer ({})...'.format(model_name))
	# Load tokenizer for pretrained model
	tokenizer = AutoTokenizer.from_pretrained(model_name)

	if lemmatization or pos_tagging:
		print("Loading spaCy ({})".format(spacy_model))
		nlp = spacy.load(spacy_model)


	tokenized_sentences = []
	lemmatized_sentences = []
	pos_tagged_sentences = []
	print('Tokenizing sentences...')
	misaligned_count = 0
	for i in tqdm(range(len(sentences)), mininterval=30):
		sentence_dct = json.loads(sentences[i])
		text = sentence_dct['text'].strip()
		text = re.sub('#+', '#', text)
		text = re.sub('_', ' ', text)
		text = re.sub('▁', ' ', text)
		text = ''.join(ch for ch in text if ch not in string.punctuation)

		if len(text) > 0:
			tokens = tokenizer(text, add_special_tokens=False).tokens()
			if len(tokens) == 0:
				continue

			common_format_tokens = []
			current_token = [clean_tokenizer_formatting(tokens[0], model_name)]
			for token in tokens[1:]:
				if is_token_piece(token, model_name): 
					current_token.append(clean_tokenizer_formatting(token, model_name))
				else:
					common_format_tokens.append(current_token)
					current_token = [clean_tokenizer_formatting(token, model_name)]
			common_format_tokens.append(current_token)

			combined_tokens = [''.join(l) for l in common_format_tokens]
			combined_tokens_string = ' '.join(combined_tokens)
				
			tokenized_sentences.append({'id': sentence_dct['id'],
										'source': sentence_dct['source'],
										'tokens': common_format_tokens,
										'text': text
										})
			
			if lemmatization or pos_tagging:
				# cleaned_tokens = [re.sub('_', '', re.sub('##', '', token)) for token in word_pieces_combined_tokens]
				# clean_tokens_string = ' '.join(cleaned_tokens)
				processed_doc = nlp(combined_tokens_string)
				spacy_tokens = []
				lemmas = []
				pos_tags = []
				for spacy_token in processed_doc:
					spacy_tokens.append(spacy_token)
					lemmas.append(spacy_token.lemma_)
					pos_tags.append(spacy_token.pos_)
				
				aligned = False
				try:
					assert len(combined_tokens) == len(spacy_tokens)
					assert len(combined_tokens) == len(lemmas)
					assert len(combined_tokens) == len(pos_tags)
					aligned = True
				except AssertionError as e:
					misaligned_count += 1
					len_1 = len(combined_tokens)
					len_2 = len(spacy_tokens)
					if verbose:
						print('Sentence:', text)
						for i in range(max(len_1, len_2)):
							if i >= len_1:
								print('None', spacy_tokens[i].text, lemmas[i], pos_tags[i])
							elif i >= len_2:
								print(combined_tokens[i], 'None', 'None', 'None')
							else:
								print(combined_tokens[i], spacy_tokens[i].text, lemmas[i], pos_tags[i])
						print('-'*75)

				if lemmatization:
					lemmatized_sentences.append({'id': sentence_dct['id'],
												 'source': sentence_dct['source'],
												 'lemmas': lemmas,
												 'aligned': aligned})
				if pos_tagging:
					pos_tagged_sentences.append({'id': sentence_dct['id'],
												 'source': sentence_dct['source'],
												 'pos_tags': pos_tags,
												 'aligned': aligned})

	print('Sentences with misaligned tokens and lemmas/part-of-speech tags: {}'.format(misaligned_count))
	print()

	tokenized_outfile = os.path.join(outdir, 'tokenized_all.jsonlist')
	print('Writing n={} tokenized sentences into {} file'.format(len(tokenized_sentences), tokenized_outfile))
	with open(tokenized_outfile, 'w') as f:
		for sent in tokenized_sentences:
			f.write(json.dumps(sent)+'\n')

	if lemmatization:
		lemmatized_outfile = os.path.join(outdir, 'lemmatized_all.jsonlist')
		print('Writing lemmatized sentences into {} file'.format(lemmatized_outfile))
		with open(lemmatized_outfile, 'w') as f:
			for sent in lemmatized_sentences:
				f.write(json.dumps(sent)+'\n')

	if pos_tagging:
		pos_tagged_outfile = os.path.join(outdir, 'pos_tagged_all.jsonlist')
		print('Writing pos-tagged sentences into {} file'.format(pos_tagged_outfile))
		with open(pos_tagged_outfile, 'w') as f:
			for sent in pos_tagged_sentences:
				f.write(json.dumps(sent)+'\n')


if __name__ == '__main__':
	main()















