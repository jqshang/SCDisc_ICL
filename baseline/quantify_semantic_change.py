import json
import pickle
import os
from tqdm import tqdm
from optparse import OptionParser

from misc_utils import *
from evaluation_utils import *



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


	# Load data files
	with open('../data/{}/targets.json'.format(dataset)) as f:
		targets_to_true_score = json.load(f)

	with open('../data/{}/processed_{}/target_indices.json'.format(dataset, model_name)) as f:
		target_indices = json.load(f)

	with open('../data/{}/processed_{}/control_indices.json'.format(dataset, model_name)) as f:
		control_indices = json.load(f)


	# Get term occurrence sentence ids by period
	term_sent_ids_by_period = {}
	for term in target_indices:
		term_sent_ids_by_period[term] = {}
		for line_id, period, _, _ in target_indices[term]:
			if period not in term_sent_ids_by_period[term]:
				term_sent_ids_by_period[term][period] = []
			term_sent_ids_by_period[term][period].append(line_id)

	for term in control_indices:
		term_sent_ids_by_period[term] = {}
		for line_id, period, _, _ in control_indices[term]:
			if period not in term_sent_ids_by_period[term]:
				term_sent_ids_by_period[term][period] = []
			term_sent_ids_by_period[term][period].append(line_id)



	# ----------------------------------------- #
	#          QUANTIFY SEMANTIC CHANGE         #
	# ----------------------------------------- #

	raw_change_fname = '../results/semantic_change_scores/{}__{}__raw_scores_by_method.json'.format(dataset.lower(), model_name)
	print('# ----------------------------------------- #')
	print('#          QUANTIFY SEMANTIC CHANGE         #')
	print('# ----------------------------------------- #')

	raw_semantic_change_scores_by_method = {'emb_prt': {},
											'emb_apd': {},
											'subst_jsd': {},
											'clustr_ap_wd': {},
											'clustr_k5_wd': {}}


	# EMBEDDINGS-BASED REPRESENTATIONS

	# Load APD scores
	with open('../results/semantic_change_scores/{}__{}__avg_pairwise_dist_by_term.json'.format(dataset.lower(), model_name)) as f:
		avg_pairwise_dist_by_term = json.load(f)

	print('\n Compute semantic change using embeddings (PRT and APD)...')
	
	errors = []
	embeddings_dir = '../representations/{}__{}/target_embeddings'.format(dataset.lower(), model_name)
	for term in tqdm(target_indices, desc='Emb (PRT & APD) -- target', mininterval=60):
		if len(term_sent_ids_by_period[term]) < 2:
			continue
		if not os.path.isfile(os.path.join(embeddings_dir, term+'_embeddings.pickle')):
			errors.append(term)
			continue
		raw_semantic_change_scores_by_method['emb_prt'][term] = compute_emb_prt_change_score(term, 
																							 term_sent_ids_by_period[term][period_1_str], 
																							 term_sent_ids_by_period[term][period_2_str], 
																							 dataset, model_name,
																							 is_target_term=True)
		raw_semantic_change_scores_by_method['emb_apd'][term] = avg_pairwise_dist_by_term[term]

	embeddings_dir = '../representations/{}__{}/control_embeddings'.format(dataset.lower(), model_name)
	for term in tqdm(control_indices, desc='Emb (PRT & APD) -- control', mininterval=60):
		if len(term_sent_ids_by_period[term]) < 2:
			continue
		if not os.path.isfile(os.path.join(embeddings_dir, term+'_embeddings.pickle')):
			errors.append(term)
			continue
		raw_semantic_change_scores_by_method['emb_prt'][term] = compute_emb_prt_change_score(term, 
																							 term_sent_ids_by_period[term][period_1_str], 
																							 term_sent_ids_by_period[term][period_2_str],
																							 dataset, model_name,
																							 is_target_term=False)
		raw_semantic_change_scores_by_method['emb_apd'][term] = avg_pairwise_dist_by_term[term]

	print('# of errors in obtaining representations:', len(errors))
	
	emb_prt_corr = correlation(targets_to_true_score, raw_semantic_change_scores_by_method['emb_prt'])
	emb_apd_corr = correlation(targets_to_true_score, raw_semantic_change_scores_by_method['emb_apd'])
	
	print('EMB (PRT) correlation:', round(emb_prt_corr, 3))
	print('EMB (APD) correlation:', round(emb_apd_corr, 3))
	
	print('\n'+'_'*75)


	# SUBSTITUTES-BASED REPRESENTATIONS

	print('\n Compute semantic change using MLM substitutes (JSD)...')

	errors = []
	substitutes_dir = '../representations/{}__{}/target_substitutes'.format(dataset.lower(), model_name)
	for term in tqdm(target_indices, desc='Subst (JSD) -- target', mininterval=60):
		if len(term_sent_ids_by_period[term]) < 2:
			continue
		if not os.path.isfile(os.path.join(substitutes_dir, term+'_substitutes.pickle')):
			errors.append(term)
			continue
		raw_semantic_change_scores_by_method['subst_jsd'][term] = compute_subs_jsd_change_score(term, 
																								term_sent_ids_by_period[term][period_1_str], 
																								term_sent_ids_by_period[term][period_2_str],
																								dataset, model_name,
																								is_target_term=True)
	
	substitutes_dir = '../representations/{}__{}/control_substitutes'.format(dataset.lower(), model_name)
	for term in tqdm(control_indices, desc='Subst (JSD) -- control', mininterval=60):
		if len(term_sent_ids_by_period[term]) < 2:
			continue
		if not os.path.isfile(os.path.join(substitutes_dir, term+'_substitutes.pickle')):
			errors.append(term)
			continue
		raw_semantic_change_scores_by_method['subst_jsd'][term] = compute_subs_jsd_change_score(term, 
																								term_sent_ids_by_period[term][period_1_str], 
																								term_sent_ids_by_period[term][period_2_str],
																								dataset, model_name,
																								is_target_term=False)
	
	print('# of errors in obtaining representations:', len(errors))

	subst_jsd_corr = correlation(targets_to_true_score, raw_semantic_change_scores_by_method['subst_jsd'])

	print('EMB (APD) correlation:', round(subst_jsd_corr, 3))
	
	print('\n'+'_'*75)	


	# CLUSTERS-BASED REPRESENTATIONS
	agg_type_1 = 'AP period_1-period_2'
	agg_type_2 = 'K5 period_1-period_2'
	results_path = '../representations/{}__{}/montariol_results'.format(dataset.lower(), model_name)
	for f_name in tqdm(os.listdir(results_path), desc='Clustr (AP & K5) -- target & control', mininterval=60):
		if not f_name.endswith('json'):
			continue
		with open(os.path.join(results_path, f_name)) as f:
			word_dct = json.load(f)
			if len(word_dct) > 0:
				word = word_dct['word']
				raw_semantic_change_scores_by_method['clustr_ap_wd'][word] = word_dct[agg_type_1]
				raw_semantic_change_scores_by_method['clustr_k5_wd'][word] = word_dct[agg_type_2]
	errors = []
	for term in list(target_indices.keys())+list(control_indices.keys()):
		if term not in raw_semantic_change_scores_by_method['clustr_ap_wd']:
			errors.append(term)

	print('# of errors in obtaining representations:', len(errors))


	# SAVE ALL RESULTS
	with open(raw_change_fname, 'w') as f:
		json.dump(raw_semantic_change_scores_by_method, f)



if __name__ == '__main__':
	main()

