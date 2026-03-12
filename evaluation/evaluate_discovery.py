import json
import pickle
import os
from tqdm import tqdm
from optparse import OptionParser

from misc_utils import *
from evaluation_utils import *



# ----------------------------------------- #
#              HELPER FUNCTIONS             #
# ----------------------------------------- #

def evaluate_discovery_base(method):
	terms_ranked_by_change_score = sorted([term
										   for term in W_list],
										  key=lambda t: raw_semantic_change_scores_by_method[method].get(t, 0),
										  reverse=True
										 )
	print('\n'+'_'*75 + '\n')
	print('Method (base):', method)
	for top_x in top_x_for_printing:
		discovered_count = len(set(terms_ranked_by_change_score[:top_x]).intersection(set(T_star_list)))
		print('top_x={}: {} ({} %)'.format(top_x, 
										   discovered_count, 
										   round(discovered_count/len(T_star_list)*100, 2)))
	return [W_list_mapping[term] for term in terms_ranked_by_change_score]


def evaluate_discovery_PT(method):
	if method == 'emb_prt':
		perm_pvals = emb_prt_perm_pvals
	elif method == 'emb_apd':
		perm_pvals = emb_apd_perm_pvals
	elif method == 'subst_jsd':
		perm_pvals = subst_jsd_perm_pvals
	else:
		raise ValueError('Permutation tests were only run on emb_prt, emb_apd, and subst_jsd')
	terms_ranked_by_change_score = sorted([elt for elt in W_list
										   if elt in perm_pvals 
										   and perm_pvals[elt]<=0.05],
										   key=lambda t: raw_semantic_change_scores_by_method[method].get(t, 0),
										   reverse=True)
	print('\n'+'_'*75 + '\n')
	print('Method (+ PT):', method)
	for top_x in top_x_for_printing:
		discovered_count = len(set(terms_ranked_by_change_score[:top_x]).intersection(set(T_star_list)))
		print('top_x={}: {} ({} %)'.format(top_x, 
										   discovered_count, 
										   round(discovered_count/len(T_star_list)*100, 2)))
	return [W_list_mapping[term] for term in terms_ranked_by_change_score]


def evaluate_discovery_PT_FDR(method):
	if method == 'emb_prt':
		perm_pvals = emb_prt_perm_pvals
	elif method == 'emb_apd':
		perm_pvals = emb_apd_perm_pvals
	elif method == 'subst_jsd':
		perm_pvals = subst_jsd_perm_pvals
	else:
		raise ValueError('Permutation tests were only run on emb_prt, emb_apd, and subst_jsd')
	terms_ranked_by_change_score = sorted([elt for elt in W_list
										   if elt in perm_pvals 
										   and perm_pvals[elt]==0],
										   key=lambda t: raw_semantic_change_scores_by_method[method].get(t, 0),
										   reverse=True)
	print('\n'+'_'*75 + '\n')
	print('Method (+ PT + FDR):', method)
	for top_x in top_x_for_printing:
		discovered_count = len(set(terms_ranked_by_change_score[:top_x]).intersection(set(T_star_list)))
		print('top_x={}: {} ({} %)'.format(top_x, 
										   discovered_count, 
										   round(discovered_count/len(T_star_list)*100, 2)))
	return [W_list_mapping[term] for term in terms_ranked_by_change_score]


def evaluate_discovery_FS(method):
	if 'freq_scaled_dist__{}'.format(method) not in scaled_semantic_change_scores_by_method:
		scaled_semantic_change_scores_by_method['freq_scaled_dist__{}'.format(method)] = {}
		for term in tqdm(W_list, desc='computing scaled change scores', mininterval=60):
			if term not in set_T_per_term or term not in raw_semantic_change_scores_by_method[method]:
				continue
			if term not in scaled_semantic_change_scores_by_method['freq_scaled_dist__{}'.format(method)]:
				scaled_change_score = compute_scaled_score(term,
														   raw_semantic_change_scores_by_method[method],
														   set_T_per_term[term])
				scaled_semantic_change_scores_by_method['freq_scaled_dist__{}'.format(method)][term] = scaled_change_score
	terms_ranked_by_change_score = sorted([term for term in W_list if
										  term in scaled_semantic_change_scores_by_method['freq_scaled_dist__{}'.format(method)]],
										 key=lambda t: 
										 scaled_semantic_change_scores_by_method['freq_scaled_dist__{}'.format(method)][t],
										 reverse=True)
	for top_x in top_x_for_printing:
		discovered_count = len(set(terms_ranked_by_change_score[:top_x]).intersection(set(T_star_list)))
		print('top_x={}: {} ({} %)'.format(top_x, 
										   discovered_count, 
										   round(discovered_count/len(T_star_list)*100, 2)))
	return [W_list_mapping[term] for term in terms_ranked_by_change_score]


def evaluate_discovery_FS_PM(method):
	if 'pos_matched_freq_scaled_dist__{}'.format(method) not in scaled_semantic_change_scores_by_method:
		scaled_semantic_change_scores_by_method['pos_matched_freq_scaled_dist__{}'.format(method)] = {}
		for term in tqdm(W_list, desc='computing scaled change scores', mininterval=60):
			if term not in set_T_per_term or term not in raw_semantic_change_scores_by_method[method]:
				continue
			if term not in scaled_semantic_change_scores_by_method['pos_matched_freq_scaled_dist__{}'.format(method)]:
				scaled_change_score = compute_scaled_score(term,
														   raw_semantic_change_scores_by_method[method],
														   set_T_per_term[term])
				scaled_semantic_change_scores_by_method['pos_matched_freq_scaled_dist__{}'.format(method)][term] = scaled_change_score
	terms_ranked_by_change_score = sorted([term for term in W_list if
										  term in raw_semantic_change_scores_by_method[method] and
										  len(set_T_per_term.get(term, []))>0],
										 key=lambda t: 
										 scaled_semantic_change_scores_by_method['pos_matched_freq_scaled_dist__{}'.format(method)][t],
										 reverse=True)
	for top_x in top_x_for_printing:
		discovered_count = len(set(terms_ranked_by_change_score[:top_x]).intersection(set(T_star_list)))
		print('top_x={}: {} ({} %)'.format(top_x, 
										   discovered_count, 
										   round(discovered_count/len(T_star_list)*100, 2)))
	return [W_list_mapping[term] for term in terms_ranked_by_change_score]




# ----------------------------------------- #
#                MAIN FUNCTION              #
# ----------------------------------------- #

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
	parser.add_option('--do-discovery-base', 
					  action='store_true',
					  help='Evaluate discovery using base methods: default=%default')
	parser.add_option('--do-discovery-FS', 
					  action='store_true',
					  help='Evaluate discovery using frequency scaling: default=%default')
	parser.add_option('--do-discovery-FS-PM', 
					  action='store_true',
					  help='Evaluate discovery using frequency scaling and part-of-speech matching: default=%default')
	parser.add_option('--do-discovery-PT', 
					  action='store_true',
					  help='Evaluate discovery using permutation tests: default=%default')
	parser.add_option('--do-discovery-PT-FDR', 
					  action='store_true',
					  help='Evaluate discovery using premutation tests and false-discovery-rate control: default=%default')



	(options, args) = parser.parse_args()
	dataset = options.dataset

	if dataset == 'LiverpoolFC':
		period_1_str = 'period_2011-13'
		period_2_str = 'period_2017'
		threshold_beta = 0.59
	elif dataset == 'semeval_en':
		period_1_str = 'corpus1'
		period_2_str = 'corpus2'
		threshold_beta = 0.4
	else:
		period_1_str = 'period_1'
		period_2_str = 'period_2'

	model = options.model
	model_name = extract_model_name_from_path(model)
	
	do_discovery_base = options.do_discovery_base
	do_discovery_FS = options.do_discovery_FS
	do_discovery_FS_PM = options.do_discovery_FS_PM
	do_discovery_PT = options.do_discovery_PT
	do_discovery_PT_FDR = options.do_discovery_PT_FDR

	assert do_discovery_base or do_discovery_FS or do_discovery_FS_PM or do_discovery_PT or do_discovery_PT_FDR, 'At least one discovery evalutations should be enabled...'

	# Load data files
	with open('../data/{}/processed_{}/stats/token_by_pos_frequencies.json'.format(dataset, model_name)) as f:
		token_by_pos_freq_dct = json.load(f)

	with open('../data/{}/targets.json'.format(dataset)) as f:
		targets_to_true_score = json.load(f)

	with open('../data/{}/processed_{}/control_indices.json'.format(dataset, model_name)) as f:
		control_indices = json.load(f)
	control_terms = list(control_indices.keys())


	# ----------------------------------------- #
	#   Semantic Change Discovery Experiments   #
	# ----------------------------------------- #

	print('# ----------------------------------------- #')
	print('#   Semantic Change Discovery Experiments   #')
	print('# ----------------------------------------- #')

	global raw_semantic_change_scores_by_method
	with open('../results/semantic_change_scores/{}__{}__raw_scores_by_method.json'.format(dataset.lower(), model_name)) as f:
		raw_semantic_change_scores_by_method = json.load(f)

	global scaled_semantic_change_scores_by_method
	if os.path.isfile('../results/semantic_change_scores/{}__{}__scaled_scores_by_method.json'.format(dataset.lower(), model_name)):
		with open('../results/semantic_change_scores/{}__{}__scaled_scores_by_method.json'.format(dataset.lower(), model_name)) as f:
			scaled_semantic_change_scores_by_method = json.load(f)
	else:
		scaled_semantic_change_scores_by_method = {}

	if len(scaled_semantic_change_scores_by_method) < 10:
		do_compute_scaled_change = True

	# (based on experiments) scaling factors were selected to maximize the average 
	# correlation score (i.e., semantic change detection performance) across methods 
	scaling_factors = {'semeval_en': {'bert-base-uncased': (2.5, 1.9),
									  'bert-base-multilingual-uncased': (1.5, 1.6),
									  'xlm-roberta-base': (1.5, 2.3),
									  'xl-lexeme': (2.3, 2.1)},
					   'LiverpoolFC': {'bert-base-uncased': (2.2, 2.4),
									   'bert-base-multilingual-uncased': (2.5, 2.5),
									   'xlm-roberta-base': (2.5, 2.5),
									   'xl-lexeme': (2.2, 2.5)}}

	global emb_prt_perm_pvals
	with open('../results/permutations/{}__{}__emb_prt_permutation_pvals.json'.format(dataset.lower(), model_name)) as f:
		emb_prt_perm_pvals = json.load(f)

	global emb_apd_perm_pvals
	with open('../results/permutations/{}__{}__emb_apd_permutation_pvals.json'.format(dataset.lower(), model_name)) as f:
		emb_apd_perm_pvals = json.load(f)

	global subst_jsd_perm_pvals
	with open('../results/permutations/{}__{}__subst_jsd_permutation_pvals.json'.format(dataset.lower(), model_name)) as f:
		subst_jsd_perm_pvals = json.load(f)

	all_top_k_ranked_sets = {}
	global top_x_for_printing 
	top_x_for_printing = [100, 150, 250, 500, 1000, 2500, 5000, 10000]
	
	
	# Define T* list
	global T_star_list
	T_star_list = [term.split('_')[0] for term in targets_to_true_score if targets_to_true_score[term]>threshold_beta]
	global W_list
	W_list = list(set(control_terms).difference(set(T_star_list))) + T_star_list
	global W_list_mapping
	W_list_mapping = {}
	global W_list_inverse_mapping
	W_list_inverse_mapping = {}
	for term in sorted(W_list):
		W_list_inverse_mapping[len(W_list_mapping)] = term
		W_list_mapping[term] = len(W_list_mapping)

	print('Dataset:', dataset)
	print('Model:', model_name)
	print('len(T_star_list) =', len(T_star_list))
	print('len(W_list) =', len(W_list))
	print('T_star_list =', T_star_list)


	if do_discovery_base:
		# EMB (PRT)
		method = 'emb_prt'
		all_top_k_ranked_sets['{}__base'.format(method)] = evaluate_discovery_base(method)

		# EMB (APD)
		method = 'emb_apd'
		all_top_k_ranked_sets['{}__base'.format(method)] = evaluate_discovery_base(method)


		# SUBST (JSD)
		method = 'subst_jsd'
		all_top_k_ranked_sets['{}__base'.format(method)] = evaluate_discovery_base(method)

		
		# CLUSTR (AP)
		method = 'clustr_ap_wd'
		all_top_k_ranked_sets['{}__base'.format(method)] = evaluate_discovery_base(method)


		# CLUSTR (K5)
		method = 'clustr_k5_wd'
		all_top_k_ranked_sets['{}__base'.format(method)] = evaluate_discovery_base(method)


	if do_discovery_PT:
		# EMB (PRT) + PT
		method = 'emb_prt'
		all_top_k_ranked_sets['{}__PT'.format(method)] = evaluate_discovery_PT(method)


		# EMB (APD) + PT
		method = 'emb_apd'
		all_top_k_ranked_sets['{}__PT'.format(method)] = evaluate_discovery_PT(method)


		# SUBST (JSD) + PT
		method = 'subst_jsd'
		all_top_k_ranked_sets['{}__PT'.format(method)] = evaluate_discovery_PT(method)

	if do_discovery_PT_FDR:
		# EMB (PRT) + PT + FDR
		method = 'emb_prt'
		all_top_k_ranked_sets['{}__PT_FDR'.format(method)] = evaluate_discovery_PT_FDR(method)


		# EMB (APD) + PT + FDR
		method = 'emb_apd'
		all_top_k_ranked_sets['{}__PT_FDR'.format(method)] = evaluate_discovery_PT_FDR(method)


		# SUBST (JSD) + PT + FDR
		method = 'subst_jsd'
		all_top_k_ranked_sets['{}__PT_FDR'.format(method)] = evaluate_discovery_PT_FDR(method)


	global set_T_per_term
	if do_discovery_FS:
		if do_compute_scaled_change:
			# PREPARE SET_T FOR (FS)
			scaling_factor = scaling_factors[dataset][model_name][0]
			set_T_per_term = {}
			for term_x in tqdm(W_list, mininterval=60):
				term_x_freq = sum(token_by_pos_freq_dct[term_x].values())
				set_T_per_term[term_x] = []
				for term_t in W_list:
					if term_t != term_x:
						term_t_freq = sum(token_by_pos_freq_dct[term_t].values())
						if term_t_freq <= scaling_factor*term_x_freq and term_t_freq >= term_x_freq/scaling_factor:
							set_T_per_term[term_x].append(term_t)

		# EMB (PRT) + FS
		method = 'emb_prt'
		all_top_k_ranked_sets['{}__FS'.format(method)] = evaluate_discovery_FS(method)

		# EMB (APD) + FS
		method = 'emb_apd'
		all_top_k_ranked_sets['{}__FS'.format(method)] = evaluate_discovery_FS(method)


		# SUBST (JSD) + FS
		method = 'subst_jsd'
		all_top_k_ranked_sets['{}__FS'.format(method)] = evaluate_discovery_FS(method)

		
		# CLUSTR (AP) + FS
		method = 'clustr_ap_wd'
		all_top_k_ranked_sets['{}__FS'.format(method)] = evaluate_discovery_FS(method)


		# CLUSTR (K5) + FS
		method = 'clustr_k5_wd'
		all_top_k_ranked_sets['{}__FS'.format(method)] = evaluate_discovery_FS(method)


		# SAVE COMPUTED SCALED CHANGE
		if do_compute_scaled_change:
			with open('../results/semantic_change_scores/{}__{}__scaled_scores_by_method.json'.format(dataset.lower(), model_name), 'w') as f:
				json.dump(scaled_semantic_change_scores_by_method, f)


	if do_discovery_FS_PM:
		if do_compute_scaled_change:
			# PREPARE SET_T FOR (FS + PM)
			scaling_factor = scaling_factors[dataset][model_name][1]
			set_T_per_term = {}
			for term_x in tqdm(W_list, mininterval=60):
				term_x_freq = sum(token_by_pos_freq_dct[term_x].values())
				term_x_pos = determine_dominant_pos(term_x, token_by_pos_freq_dct)
				if term_x_pos is None:
					continue
				set_T_per_term[term_x] = []
				for term_t in W_list:
					if term_t != term_x:
						term_t_freq = sum(token_by_pos_freq_dct[term_t].values())
						term_t_pos = determine_dominant_pos(term_t, token_by_pos_freq_dct)
						if term_x_pos == term_t_pos:
							if term_t_freq <= scaling_factor*term_x_freq and term_t_freq >= term_x_freq/scaling_factor:
								set_T_per_term[term_x].append(term_t)

		# EMB (PRT) + FS + PM
		method = 'emb_prt'
		all_top_k_ranked_sets['{}__FS_PM'.format(method)] = evaluate_discovery_FS_PM(method)

		# EMB (APD) + FS + PM
		method = 'emb_apd'
		all_top_k_ranked_sets['{}__FS_PM'.format(method)] = evaluate_discovery_FS_PM(method)


		# SUBST (JSD) + FS + PM
		method = 'subst_jsd'
		all_top_k_ranked_sets['{}__FS_PM'.format(method)] = evaluate_discovery_FS_PM(method)

		
		# CLUSTR (AP) + FS + PM
		method = 'clustr_ap_wd'
		all_top_k_ranked_sets['{}__FS_PM'.format(method)] = evaluate_discovery_FS_PM(method)


		# CLUSTR (K5) + FS + PM
		method = 'clustr_k5_wd'
		all_top_k_ranked_sets['{}__FS_PM'.format(method)] = evaluate_discovery_FS_PM(method)


		# SAVE COMPUTED SCALED CHANGE
		if do_compute_scaled_change:
			with open('../results/semantic_change_scores/{}__{}__scaled_scores_by_method.json'.format(dataset.lower(), model_name), 'w') as f:
				json.dump(scaled_semantic_change_scores_by_method, f)



	# ------------------------------------------ #
	#  FILTER RANKINGS & SAVE DISCOVERY RESULTS  #
	# ------------------------------------------ #

	print('# ------------------------------------------ #')
	print('#  FILTER RANKINGS & SAVE DISCOVERY RESULTS  #')
	print('# ------------------------------------------ #')
	filtered_term_rankings_by_method = {}
	for method in tqdm(all_top_k_ranked_sets):
		ranking = all_top_k_ranked_sets[method]
		filtered_ranking = []
		for term_i in ranking:
			term = W_list_inverse_mapping[term_i]
			if (term in targets_to_true_score or determine_dominant_pos_non_combined(term, token_by_pos_freq_dct, threshold=0.5) in ['NOUN', 'VERB', 'ADJ']):
				filtered_ranking.append(term_i)
		filtered_term_rankings_by_method[method] = filtered_ranking


	filtered_W_list = [term for term in W_list
						if term in targets_to_true_score or 
						determine_dominant_pos_non_combined(term, token_by_pos_freq_dct, threshold=0.5) 
						in ['NOUN', 'VERB', 'ADJ']]


	# SAVE DISCOVERY RESULTS

	with open('../results/rankings/{}__{}__W_list_mapping.json'.format(dataset.lower(), model_name), 'w') as f:
		json.dump(W_list_mapping, f)

	with open('../results/rankings/{}__{}__W_list_inverse_mapping.json'.format(dataset.lower(), model_name), 'w') as f:
		json.dump(W_list_inverse_mapping, f)
	
	for method in all_top_k_ranked_sets:
		with open('../results/rankings/{}__{}__ranked_terms__{}.json'.format(dataset.lower(), model_name, method), 'w') as f:
			json.dump(all_top_k_ranked_sets[method], f)

		with open('../results/rankings/{}__{}__filtered_ranked_terms__{}.json'.format(dataset.lower(), model_name, method), 'w') as f:
			json.dump(filtered_term_rankings_by_method[method], f)



	# ----------------------------------------- #
	#           COMPUTE AVERAGE RANK            #
	# ----------------------------------------- #

	print('# ----------------------------------------- #')
	print('#           COMPUTE AVERAGE RANK            #')
	print('# ----------------------------------------- #')

	avg_rank_by_method = {}
	filtered_avg_rank_by_method = {}
	T_star_rankings_by_method = {}
	filtered_T_star_rankings_by_method = {}
	for method in all_top_k_ranked_sets:
		ranked_list = all_top_k_ranked_sets[method]
		filtered_ranked_list = filtered_term_rankings_by_method[method]
		N, filtered_N = len(W_list), len(filtered_W_list)
		ranks, filtered_ranks = [], []
		for term in T_star_list[::-1]:
			term_found = False
			for i in range(len(ranked_list)):
				if term == W_list_inverse_mapping[ranked_list[i]]:
					term_found = True
					break
			if term_found:
				ranks.append(i)
			else:
				ranks.append(N)
			
			term_found = False
			for i in range(len(filtered_ranked_list)):
				if term == W_list_inverse_mapping[filtered_ranked_list[i]]:
					term_found = True
					break
			if term_found:
				filtered_ranks.append(i)
			else:
				filtered_ranks.append(filtered_N)
		avg_rank_by_method[method] = sum(ranks)/len(ranks)
		filtered_avg_rank_by_method[method] = sum(filtered_ranks)/len(filtered_ranks)
		T_star_rankings_by_method[method] = ranks
		filtered_T_star_rankings_by_method[method] = filtered_ranks
		print(method)
		print('(filtered) Average rank:', filtered_avg_rank_by_method[method])
		print('(filtered) Rankings:', filtered_T_star_rankings_by_method[method])


	# Save results
	for method in all_top_k_ranked_sets:
		with open('../results/average_rank/{}__{}__T_star_ranking__{}.json'.format(dataset.lower(), model_name, method), 'w') as f:
			json.dump({'average_rank': avg_rank_by_method[method],
					   'rankings': T_star_rankings_by_method[method]}, f)

		with open('../results/average_rank/{}__{}__filtered_T_star_ranking__{}.json'.format(dataset.lower(), model_name, method), 'w') as f:
			json.dump({'average_rank': filtered_avg_rank_by_method[method],
					   'rankings': filtered_T_star_rankings_by_method[method]}, f)


if __name__ == '__main__':
	main()

