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
					if type(embeddings_dct[line_id_1]) == list:
						point_wise_distances.append(cosine(embeddings_dct[line_id_1][0], embeddings_dct[line_id_2][0]))
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
	parser.add_option('--terms-start-id',
						type=int,
						default=0,
						help='First id of terms to process: default=%default')
	parser.add_option('--terms-end-id',
						type=int,
						default=-1,
						help='Last id of terms (inclusive) to process: default=%default')

	(options, args) = parser.parse_args()

	dataset = options.dataset
	model = options.model
	model_name = extract_model_name_from_path(model)
	terms_start_id = options.terms_start_id
	terms_end_id = options.terms_end_id

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

	print('RANK CHANGES BY APD')
	with open("../results/semantic_change_scores/{}__{}__avg_pairwise_dist_by_term.json".format(dataset.lower(), model_name)) as f:
		avg_pairwise_dist_by_term = json.load(f)

	terms_ranked_by_change_score = sorted(list(avg_pairwise_dist_by_term.keys()),
											key=lambda t: (t in target_terms, avg_pairwise_dist_by_term[t]),
											reverse=True)

	print('RUN PERMUTATIONS')
	embedding_representation_perm_pvals = {}
	if terms_end_id == -1:
		select_terms = terms_ranked_by_change_score[terms_start_id:]
	else:
		select_terms = terms_ranked_by_change_score[terms_start_id:terms_end_id+1]
	for term in tqdm(select_terms):
		try:
			if term in target_terms:
				is_target_term = True
			else:
				is_target_term = False

			embeddings_dir = '../representations/{}__{}/{}_embeddings'.format(dataset.lower(), model_name, 'target' if is_target_term else 'control')
			if not os.path.exists(os.path.join(embeddings_dir, term+'_embeddings.pickle')):
				continue
			with open(os.path.join(embeddings_dir, term+'_embeddings.pickle'), 'rb') as f:
				embeddings_dct = pickle.load(f)
			
			all_pairwise_dists = {}
			period_1_term_sent_ids = term_sent_ids_by_period[term][time_periods[0]]
			random.shuffle(period_1_term_sent_ids)
			period_2_term_sent_ids = term_sent_ids_by_period[term][time_periods[1]]
			random.shuffle(period_2_term_sent_ids)
			all_sentence_ids = period_1_term_sent_ids[:100] + period_2_term_sent_ids[:100]	
			random.shuffle(all_sentence_ids)

			n = len(all_sentence_ids)
			r = 100
			# r = min(len(term_sent_ids_by_period[term][time_periods[0]]), 
			# 		len(term_sent_ids_by_period[term][time_periods[1]]))
			# all_combos_count = min(count_combinations(n, r), 1e6)
			
			term_change_score = avg_pairwise_dist_by_term[term]
			change_distribution = []
			stop_count = 1e3
			for select_group in itertools.combinations(range(n), r):
				group_1_sent_ids = [all_sentence_ids[i] for i in select_group]
				group_2_sent_ids = [all_sentence_ids[i] for i in range(n) if i not in select_group]

				pairwise_dists = []
				for sent_id_1 in group_1_sent_ids:
					for sent_id_2 in group_2_sent_ids:
						if sent_id_1 != sent_id_2 and sent_id_1 in embeddings_dct and sent_id_2 in embeddings_dct:
							sorted_sent_id_order = sorted([sent_id_1, sent_id_2])
							sent_id_pair_name = '{}___{}'.format(sorted_sent_id_order[0], sorted_sent_id_order[1])
							if sent_id_pair_name not in all_pairwise_dists:
								if type(embeddings_dct[sent_id_1]) == list:
									all_pairwise_dists[sent_id_pair_name] = cosine(embeddings_dct[sent_id_1][0], 
																					 embeddings_dct[sent_id_2][0])
								else:
									all_pairwise_dists[sent_id_pair_name] = cosine(embeddings_dct[sent_id_1], 
																					 embeddings_dct[sent_id_2])
							pairwise_dists.append(all_pairwise_dists[sent_id_pair_name])

				change_distribution.append(sum(pairwise_dists)/len(pairwise_dists))
		
				if len(change_distribution) == 1e3:
					p_val = compute_p_value(term_change_score, change_distribution)
					if p_val < 0.05:
						stop_count = 1e4
					elif p_val < 0.005:
						stop_count = 1e5
				if len(change_distribution) == stop_count:
					break
		
			p_val = compute_p_value(term_change_score, change_distribution)
			embedding_representation_perm_pvals[term] = p_val
			if len(embedding_representation_perm_pvals) % 10 == 0:
				with open(os.path.join(outdir, '{}__{}__emb_apd_permutation_pvals.json'.format(dataset.lower(), model_name)), 'w') as f:
					json.dump(embedding_representation_perm_pvals, f)
		except Exception as err:
			print('ERROR', term, err)

	with open(os.path.join(outdir, '{}__{}__emb_apd_permutation_pvals.json'.format(dataset.lower(), model_name)), 'w') as f:
		json.dump(embedding_representation_perm_pvals, f)


if __name__ == '__main__':
	main()

