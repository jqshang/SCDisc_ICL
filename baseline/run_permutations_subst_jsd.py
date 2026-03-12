import os
import pickle
import json
from optparse import OptionParser

import itertools
import math
import random
import numpy as np
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm
from misc_utils import *


def compute_jsd_from_counters(p_counter, q_counter):
	# measure Jenson-Shannon distance for shared vocab between two sets of substitute terms    
	vocab = sorted(set(p_counter).union(q_counter))
	p_counts = np.array([p_counter.get(v, 0) for v in vocab])
	q_counts = np.array([q_counter.get(v, 0) for v in vocab])
	p_dist = p_counts / p_counts.sum()
	q_dist = q_counts / q_counts.sum()
	return jensenshannon(p_dist, q_dist, base=2)


def aggregate_substitute_representation(term, line_ids, is_target_term):
	substitutes_dir = '../representations/{}__{}/{}_substitutes'.format(dataset.lower(), model_name, 'target' if is_target_term else 'control')
	with open(os.path.join(substitutes_dir, term+'_substitutes.pickle'), 'rb') as f:
		substitutes = pickle.load(f)
	all_substitutes_counter = {}
	for line_id in line_ids:
		if line_id in substitutes: 
			for sub, _ in substitutes[line_id][:5]:
				if sub not in all_substitutes_counter:
					all_substitutes_counter[sub] = 0
				all_substitutes_counter[sub] += 1
	return all_substitutes_counter    


def compute_subs_jsd_change_score(term, period_1_line_ids, period_2_line_ids, is_target_term):
	substitutes_counter_1 = aggregate_substitute_representation(term, period_1_line_ids, is_target_term)
	substitutes_counter_2 = aggregate_substitute_representation(term, period_2_line_ids, is_target_term)
	return compute_jsd_from_counters(substitutes_counter_1, substitutes_counter_2)





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

	global dataset
	dataset = options.dataset
	model = options.model
	global model_name
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

	assert len(time_periods) == 2

	print('RANK CHANGES')
	change_scores_by_term = {}
	for term in target_terms + control_terms:
		if len(term_sent_ids_by_period[term]) < 2:
			continue
		if term in target_terms:
			is_target_term = True
		else:
			is_target_term = False


		substitutes_dir = '../representations/{}__{}/{}_substitutes'.format(dataset.lower(), model_name, 'target' if is_target_term else 'control')
		if not os.path.isfile(os.path.join(substitutes_dir, term+'_substitutes.pickle')):
			continue
		change_scores_by_term[term] = compute_subs_jsd_change_score(term, 
																	term_sent_ids_by_period[term][time_periods[0]], 
																	term_sent_ids_by_period[term][time_periods[1]],
																	is_target_term=is_target_term)

	
	terms_ranked_by_change_score = sorted(list(change_scores_by_term.keys()),
											key=lambda t: (t in target_terms, change_scores_by_term[t]),
											reverse=True)


	print('RUN PERMUTATIONS')
	substitutes_representation_perm_pvals = {}
	if terms_end_id == -1:
		select_terms = terms_ranked_by_change_score[terms_start_id:]
	else:
		select_terms = terms_ranked_by_change_score[terms_start_id:terms_end_id+1]
	for term in tqdm(select_terms):
			# try:
			if len(term_sent_ids_by_period[term]) < 2:
				continue
			if term in target_terms:
				is_target_term = True
			else:
				is_target_term = False
			
			substitutes_dir = '../representations/{}__{}/{}_substitutes'.format(dataset.lower(), model_name, 'target' if is_target_term else 'control')
			if not os.path.isfile(os.path.join(substitutes_dir, term+'_substitutes.pickle')):
				continue
			change_scores_by_term[term] = compute_subs_jsd_change_score(term, 
																		term_sent_ids_by_period[term][time_periods[0]], 
																		term_sent_ids_by_period[term][time_periods[1]],
																		is_target_term=is_target_term)

	
			all_sentence_ids = term_sent_ids_by_period[term][time_periods[0]] + term_sent_ids_by_period[term][time_periods[1]]
			random.shuffle(all_sentence_ids)
			n = len(all_sentence_ids)
			r = min(len(term_sent_ids_by_period[term][time_periods[0]]), 
					len(term_sent_ids_by_period[term][time_periods[1]]))
			all_combos_count = min(count_combinations(n, r), 1e6)
		
			change_distribution = []
			stop_count = 1e3
			for select_group in itertools.combinations(range(n), r):
				group_1_sent_ids = [all_sentence_ids[i] for i in select_group]
				group_2_sent_ids = [all_sentence_ids[i] for i in range(n) if i not in select_group]

				change_distribution.append(compute_subs_jsd_change_score(term, 
																		group_1_sent_ids,
																		group_2_sent_ids,
																		is_target_term=is_target_term))
		
				if len(change_distribution) == 1e3:
					p_val = compute_p_value(change_scores_by_term[term], change_distribution)
					if p_val < 0.05:
						stop_count = 1e4
					elif p_val < 0.005:
						stop_count = 1e5
				if len(change_distribution) == stop_count:
					break
		
			p_val = compute_p_value(change_scores_by_term[term], change_distribution)
			substitutes_representation_perm_pvals[term] = p_val
		
			if len(substitutes_representation_perm_pvals) % 100 == 0:
				with open(os.path.join(outdir, '{}__{}__subst_jsd_permutation_pvals.json'.format(dataset.lower(), model_name)), 'w') as f:
					json.dump(substitutes_representation_perm_pvals, f)
			# except:
			# 	print('ERROR: ', term)

	with open(os.path.join(outdir, '{}__{}__subst_jsd_permutation_pvals.json'.format(dataset.lower(), model_name)), 'w') as f:
		json.dump(substitutes_representation_perm_pvals, f)

if __name__ == '__main__':
	main()

