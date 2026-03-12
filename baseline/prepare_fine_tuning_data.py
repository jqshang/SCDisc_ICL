import os
import re
import json
from optparse import OptionParser
from tqdm import tqdm
import numpy as np
from misc_utils import *


def main():
	"""Scrpt to split tokenized data into training and validation sets
	in preparation for fine-tuning on MLM task.
	"""
	usage = "usage: %prog [options] arg"
	parser = OptionParser(usage)
	parser.add_option('--dataset', 
					  type=str, 
					  default='LiverpoolFC',
					  help='Dataset directory name: default=%default')
	parser.add_option('--model', 
					  type=str, 
					  default='bert-base-uncased',
					  help='Model name/path to fine-tune: default=%default')
	parser.add_option('--outdir-name',
					  type=str,
					  default='',
					  help='[optional] Manually override directory name for the fine-tuned model: default=%default')
	parser.add_option('--val_size', 
					  type=float, 
					  default=0.05,
					  help='Proportion of data that goes into the validation set: default=%default')
	parser.add_option('--seed',
					  type=int,
					  default=123,
					  help='Random seed used for training-validation splitting: default=%default')
	

	(options, args) = parser.parse_args()

	dataset = options.dataset
	model_name = extract_model_name_from_path(options.model)
	val_size = options.val_size
	seed = options.seed
	np.random.seed(seed)
	

	indir = os.path.join('../data', dataset, 'processed_' + model_name)
	infile = os.path.join(indir, 'tokenized_all.jsonlist')

	outdir = os.path.join('../models/{}__finetuned__{}'.format(dataset.lower(), model_name))
	if not os.path.exists(outdir):
		os.makedirs(outdir)

	with open(infile) as f:
		tokenized_data = f.readlines()
	print('Loaded n={} tokenized sentences'.format(len(tokenized_data)))

	training_sentences = []
	validation_sentences = []
	for line in tokenized_data:
		sent_dct = json.loads(line)
		sent_text = ' '.join([''.join(l) for l in sent_dct['tokens']])

		if val_size > 0 and np.random.rand() <= val_size:
			validation_sentences.append(sent_text)
		else:
			training_sentences.append(sent_text)

	print('Data split into training ({} sentences) and validation ({} sentences) sets'.format(len(training_sentences), len(validation_sentences)))

	if len(validation_sentences) > 0:
		sents_order = np.arange(len(validation_sentences))
		np.random.shuffle(sents_order)
		with open(os.path.join(outdir, 'all_val_sentences.txt'), 'w') as f:
			for i in sents_order:
				f.write(validation_sentences[i] + '\n')

	sents_order = np.arange(len(training_sentences))
	np.random.shuffle(sents_order)
	with open(os.path.join(outdir, 'all_train_sentences.txt'), 'w') as f:
		for i in sents_order:
			f.write(training_sentences[i] + '\n')
	
	print('Files for fine-tuning were created under:', outdir)


if __name__ == '__main__':
	main()