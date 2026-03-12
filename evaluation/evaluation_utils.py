import pickle
import os
import random
import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import cosine, jensenshannon
from tqdm import tqdm


def correlation(dct1, dct2):
    intersection = list(set(dct1.keys()).intersection(dct2.keys()))
    if len(intersection) > 1:
        data = {'var1': [dct1[term] for term in intersection],
                'var2': [dct2[term] for term in intersection]}
        df = pd.DataFrame(data)
        spearman_corr_matrix = df.corr(method='spearman')
        result = spearman_corr_matrix['var1']['var2']
        return round(float(result), 3)
    else:
        return None

def correlation_old(dct1, dct2):
    intersection = list(set(dct1.keys()).intersection(dct2.keys()))
    if len(intersection) > 1:
        result = stats.spearmanr([dct1[w] for w in intersection], [dct2[w] for w in intersection], nan_policy='omit')
        # result = stats.pearsonr([dct1[w] for w in intersection], [dct2[w] for w in intersection])
        return round(result.statistic, 3)
    else:
        return None

def compute_jsd_from_counters(p_counter, q_counter):
    # measure Jenson-Shannon distance for shared vocab between two sets of substitute terms    
    vocab = sorted(set(p_counter).union(q_counter))
    p_counts = np.array([p_counter.get(v, 0) for v in vocab])
    q_counts = np.array([q_counter.get(v, 0) for v in vocab])
    p_dist = p_counts / p_counts.sum()
    q_dist = q_counts / q_counts.sum()
    return jensenshannon(p_dist, q_dist, base=2)

def aggregate_embedding_representation(term, line_ids, dataset, model_name, is_target_term):
    embeddings_dir = '../representations/{}__{}/{}_embeddings'.format(dataset.lower(), model_name, 'target' if is_target_term else 'control')
    with open(os.path.join(embeddings_dir, term+'_embeddings.pickle'), 'rb') as f:
        embeddings = pickle.load(f)
    all_embeddings = []
    for line_id in line_ids:
        if line_id in embeddings: 
            if type(embeddings[line_id]) == list:
                all_embeddings.append(embeddings[line_id][0])
            else: 
                all_embeddings.append(embeddings[line_id])
    return np.average(all_embeddings, axis=0)

def aggregate_substitute_representation(term, line_ids, dataset, model_name, is_target_term):
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

def compute_emb_prt_change_score(term, period_1_line_ids, period_2_line_ids, dataset, model_name, is_target_term):
    agg_embedding_1 = aggregate_embedding_representation(term, period_1_line_ids, dataset, model_name, is_target_term)
    agg_embedding_2 = aggregate_embedding_representation(term, period_2_line_ids, dataset, model_name, is_target_term)
    return cosine(agg_embedding_1, agg_embedding_2)

def compute_emb_apd_change_score(term, period_1_line_ids, period_2_line_ids, dataset, model_name, is_target_term):
    embeddings_dir = '../representations/{}__{}/{}_embeddings'.format(dataset.lower(), model_name, 'target' if is_target_term else 'control')
    with open(os.path.join(embeddings_dir, term+'_embeddings.pickle'), 'rb') as f:
        embeddings = pickle.load(f)
    point_wise_distances = []
    for line_id_1 in period_1_line_ids:
        if line_id_1 in embeddings:
            for line_id_2 in period_2_line_ids:
                if line_id_2 in embeddings:
                    if type(embeddings[line_id_1]) == list:
                        point_wise_distances.append(cosine(embeddings[line_id_1][0], embeddings[line_id_2][0]))
                    else:
                        point_wise_distances.append(cosine(embeddings[line_id_1], embeddings[line_id_2]))
    return sum(point_wise_distances)/len(point_wise_distances)

def compute_subs_jsd_change_score(term, period_1_line_ids, period_2_line_ids, dataset, model_name, is_target_term):
    substitutes_counter_1 = aggregate_substitute_representation(term, period_1_line_ids, dataset, model_name, is_target_term)
    substitutes_counter_2 = aggregate_substitute_representation(term, period_2_line_ids, dataset, model_name, is_target_term)
    return compute_jsd_from_counters(substitutes_counter_1, substitutes_counter_2)
    

def compute_scaled_score(term, all_change_scores_dct, set_T):
    raw_score = all_change_scores_dct[term]
    raw_T_scores = [all_change_scores_dct[set_T[i]] 
                    for i in range(len(set_T))
                    if set_T[i] in all_change_scores_dct]
    return sum([int(raw_score >= raw_T_score) for raw_T_score in raw_T_scores])/(len(raw_T_scores)+1)

def determine_dominant_pos(term, token_by_pos_freq_dct, threshold=0.5):
    pos_tag_freqs = token_by_pos_freq_dct[term]
    term_freq = sum(token_by_pos_freq_dct[term].values())
    combined_noun_freq = pos_tag_freqs.get('PROPN', 0) + pos_tag_freqs.get('NOUN', 0)
    if combined_noun_freq > term_freq*threshold:
        return 'NOUN'
    for pos_tag in pos_tag_freqs:
        if pos_tag_freqs[pos_tag] > term_freq*threshold:
            return pos_tag
    return None

def determine_dominant_pos_non_combined(term, token_by_pos_freq_dct, threshold=0.5):
    pos_tag_freqs = token_by_pos_freq_dct[term]
    term_freq = sum(token_by_pos_freq_dct[term].values())
    for pos_tag in pos_tag_freqs:
        if pos_tag_freqs[pos_tag] > term_freq*threshold:
            return pos_tag
    return None