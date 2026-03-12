import os
from subprocess import call
from optparse import OptionParser
from misc_utils import *

# Run the MLM fine-tuning on the new dataset


def main():
	usage = "%prog "
	parser = OptionParser(usage=usage)
	parser.add_option('--dataset', 
					  type=str,
					  default='LiverpoolFC',
					  help='Dataset directory name: default=%default')
	parser.add_option('--model',
					  type=str,
					  default='bert-base-uncased',
					  help='Model to fine-tune: default=%default')
	parser.add_option('--epochs',
					  type=int,
					  default=5,
					  help='Number of epochs to fine-tune for: default=%default')
	parser.add_option('--batch-size', 
					  type=int, 
					  default=8,
					  help='Batch size: default=%default')
	parser.add_option('--seed',
					  type=int,
					  default=123,
					  help='Random seed used for fine-tuning: default=%default')

	(options, args) = parser.parse_args()

	dataset = options.dataset.lower()
	model_name = options.model
	epochs = options.epochs
	batch_size = options.batch_size
	seed = options.seed

	model_dir = '../models/{}__finetuned__{}'.format(dataset, extract_model_name_from_path(model_name))
	try:
		assert os.path.exists(model_dir)
	except AssertionError:
		print('Fine-tuned model directory is invalid/missing.')

	train_fname = os.path.join(model_dir, 'all_train_sentences.txt')
	val_fname = os.path.join(model_dir, 'all_val_sentences.txt')

	cache_dir = '../models/misc/cache'
	if not os.path.exists(cache_dir):
		os.makedirs(cache_dir)

	logs_dir = '../models/misc/logs'
	if not os.path.exists(logs_dir):
		os.makedirs(logs_dir)

	cmd = ['python',
			'-m',
			'run_mlm',
			'--model_name_or_path', model_name,
			'--cache_dir', cache_dir,
			'--train_file', train_fname,
			'--validation_file', val_fname,
			'--do_train',
			'--do_eval',
			'--max_seq_length', '256',
			'--per_device_train_batch_size', str(batch_size),
			'--per_device_eval_batch_size', str(batch_size),
			'--output_dir', model_dir,
			'--overwrite_cache',
			'--overwrite_output_dir',
			'--num_train_epochs', str(epochs),
			'--logging_dir', logs_dir,
			'--save_strategy', 'epoch',
			'--seed', str(seed)
			]

	print(cmd)
	call(cmd)

	with open(os.path.join(model_dir, 'my_cmd.txt'), 'w') as f:
		f.write(' '.join(cmd))

if __name__ == '__main__':
	main()
