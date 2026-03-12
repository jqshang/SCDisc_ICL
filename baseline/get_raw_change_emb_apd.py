import os
import pickle
import json
from optparse import OptionParser

import random
import itertools
import numpy as np
from scipy.spatial.distance import cosine

from tqdm import tqdm
from misc_utils import *


def compute_emb_apd_change_score(embeddings_dct, term, period_1_line_ids, period_2_line_ids):
	point_wise_distances = []
	for line_id_1 in period_1_line_ids:
		if line_id_1 in embeddings_dct:
			for line_id_2 in period_2_line_ids:
				if line_id_2 in embeddings_dct:
					if type(embeddings_dct[line_id_1]) == list and type(embeddings_dct[line_id_2]) == list:
						point_wise_distances.append(cosine(embeddings_dct[line_id_1][0], embeddings_dct[line_id_2][0]))
					elif type(embeddings_dct[line_id_1]) == list:
						point_wise_distances.append(cosine(embeddings_dct[line_id_1][0], embeddings_dct[line_id_2]))
					elif type(embeddings_dct[line_id_2]) == list:
						point_wise_distances.append(cosine(embeddings_dct[line_id_1], embeddings_dct[line_id_2][0]))
					else:
						point_wise_distances.append(cosine(embeddings_dct[line_id_1], embeddings_dct[line_id_2]))
	return sum(point_wise_distances)/len(point_wise_distances)

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
	parser.add_option('--continued', 
					  action='store_true',
					  help='Model to obtain mask token substitutes from: default=%default')


	(options, args) = parser.parse_args()

	dataset = options.dataset
	model = options.model
	continued = options.continued

	model_name = extract_model_name_from_path(model)

	outdir = '../results/permutations/'

	with open('../data/{}/processed_{}/target_indices.json'.format(dataset, model_name)) as f:
		target_indices = json.load(f)

	with open('../data/{}/processed_{}/control_indices.json'.format(dataset, model_name)) as f:
		control_indices = json.load(f)

	target_terms = list(target_indices.keys())
	control_terms = list(control_indices.keys())

	time_periods = []
	term_sent_ids_by_period = {}
	for term in target_indices:
		term_sent_ids_by_period[term] = {}
		for line_id, period, _, _ in target_indices[term]:
			if period not in term_sent_ids_by_period[term]:
				if period not in time_periods:
					time_periods.append(period)
				term_sent_ids_by_period[term][period] = []
			term_sent_ids_by_period[term][period].append(line_id)

	for term in control_indices:
		term_sent_ids_by_period[term] = {}
		for line_id, period, _, _ in control_indices[term]:
			if period not in term_sent_ids_by_period[term]:
				term_sent_ids_by_period[term][period] = []
			term_sent_ids_by_period[term][period].append(line_id)

	print('OBTAIN ALL PAIRWISE DISTANCES')
	if continued and os.path.isfile('../results/semantic_change_scores/{}__{}__avg_pairwise_dist_by_term.json'.format(dataset.lower(), model_name)):
		print('CONTINUING FROM AN OLD FILE FOUND')
		with open('../results/semantic_change_scores/{}__{}__avg_pairwise_dist_by_term.json'.format(dataset.lower(), model_name)) as f:
			avg_pairwise_dist_by_term = json.load(f)
	else:
		avg_pairwise_dist_by_term = {}

	# if os.path.isfile(os.path.join('../results/', 'old_{}__{}__avg_pairwise_dist_by_term.json'.format(dataset.lower(), model_name))):
	# 	with open(os.path.join('../results/', 'old_{}__{}__avg_pairwise_dist_by_term.json'.format(dataset.lower(), model_name))) as f:
	# 		avg_pairwise_dist_by_term = json.load(f)
	# 	print('OLD FILE FOUND')

	for term in tqdm(target_terms + control_terms, mininterval=100):
		# try:
			if term in avg_pairwise_dist_by_term:
				continue
			if len(term_sent_ids_by_period[term]) < 2:
				continue
			if term in target_terms:
				is_target_term = True
			else:
				is_target_term = False

			embeddings_dir = '../representations/{}__{}/{}_embeddings'.format(dataset.lower(), model_name, 'target' if is_target_term else 'control')
			if not os.path.exists(os.path.join(embeddings_dir, term+'_embeddings.pickle')):
				continue
			with open(os.path.join(embeddings_dir, term+'_embeddings.pickle'), 'rb') as f:
				embeddings_dct = pickle.load(f)
			
			avg_pairwise_dist_by_term[term] = compute_emb_apd_change_score(embeddings_dct, 
																		   term, 
																		   term_sent_ids_by_period[term][time_periods[0]], 
																		   term_sent_ids_by_period[term][time_periods[1]])
			
			if len(avg_pairwise_dist_by_term) % 10 == 0:
				with open('../results/semantic_change_scores/{}__{}__avg_pairwise_dist_by_term.json'.format(dataset.lower(), model_name), 'w') as f:
					json.dump(avg_pairwise_dist_by_term, f)
		# except:
		# 	print('ERROR', term)

	with open('../results/semantic_change_scores/{}__{}__avg_pairwise_dist_by_term.json'.format(dataset.lower(), model_name), 'w') as f:
		json.dump(avg_pairwise_dist_by_term, f)



if __name__ == '__main__':
	main()

