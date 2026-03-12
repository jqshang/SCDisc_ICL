import sys
import os
import random
import string
import math
import pickle
import itertools
import ot
import json
from collections import Counter, defaultdict
import torch
import transformers

import numpy as np

import scipy.interpolate.interpnd
from scipy.spatial.distance import cosine, cdist
from scipy import stats
from scipy.stats import entropy

# from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import KMeans

import pandas as pd

from optparse import OptionParser
from misc_utils import *
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')



def compute_averaged_embedding_dist(t1_embeddings, t2_embeddings):
    t1_mean = np.mean(t1_embeddings, axis=0, dtype=object)
    t2_mean = np.mean(t2_embeddings, axis=0, dtype=object)
    return cosine(t1_mean, t2_mean)
    # dist = 1.0 - cosine_similarity([t1_mean], [t2_mean])[0][0]
    # #print("Averaged embedding cosine dist:", dist)
    # return dist


def combine_clusters(labels, embeddings, threshold=10, remove=[]):
    cluster_embeds = defaultdict(list)
    for label, embed in zip(labels, embeddings):
        cluster_embeds[label].append(embed)
    min_num_examples = threshold
    legit_clusters = []
    for id, num_examples in Counter(labels).items():
        if num_examples >= threshold:
            legit_clusters.append(id)
        if id not in remove and num_examples < min_num_examples:
            min_num_examples = num_examples
            min_cluster_id = id

    if len(set(labels)) == 2:
        return labels

    min_dist = 1
    all_dist = []
    cluster_labels = ()
    embed_list = list(cluster_embeds.items())
    for i in range(len(embed_list)):
        for j in range(i+1,len(embed_list)):
            id, embed = embed_list[i]
            id2, embed2 = embed_list[j]
            if id in legit_clusters and id2 in legit_clusters:
                dist = compute_averaged_embedding_dist(embed, embed2)
                all_dist.append(dist)
                if dist < min_dist:
                    min_dist = dist
                    cluster_labels = (id, id2)

    std = np.std(all_dist)
    avg = np.mean(all_dist)
    limit = avg - 2 * std
    if min_dist < limit:
        for n, i in enumerate(labels):
            if i == cluster_labels[0]:
                labels[n] = cluster_labels[1]
        return combine_clusters(labels, embeddings, threshold, remove)

    if min_num_examples >= threshold:
        return labels


    min_dist = 1
    cluster_labels = ()
    for id, embed in cluster_embeds.items():
        if id != min_cluster_id:
            dist = compute_averaged_embedding_dist(embed, cluster_embeds[min_cluster_id])
            if dist < min_dist:
                min_dist = dist
                cluster_labels = (id, min_cluster_id)

    if cluster_labels[0] not in legit_clusters:
        for n, i in enumerate(labels):
            if i == cluster_labels[0]:
                labels[n] = cluster_labels[1]
    else:
        if min_dist < limit:
            for n, i in enumerate(labels):
                if i == cluster_labels[0]:
                    labels[n] = cluster_labels[1]
        else:
            remove.append(min_cluster_id)
    return combine_clusters(labels, embeddings, threshold, remove)


def cluster_word_embeddings_aff_prop(word_embeddings):
    clustering = AffinityPropagation().fit(word_embeddings)
    labels = clustering.labels_
    exemplars = clustering.cluster_centers_
    return labels, exemplars


def cluster_word_embeddings_k_means(word_embeddings, k, random_seed):
    clustering = KMeans(n_clusters=k, random_state=random_seed).fit(word_embeddings)
    labels = clustering.labels_
    exemplars = clustering.cluster_centers_
    return labels, exemplars


def compute_jsd(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    m = (p + q) / 2
    return (entropy(p, m) + entropy(q, m)) / 2


def compute_divergence_from_cluster_labels(embeds1, embeds2, labels1, labels2, threshold):
    # need to convert embeds to np arrays?
    embeds1 = np.array(embeds1)
    embeds2 = np.array(embeds2)
    
    labels_all = list(np.concatenate((labels1, labels2)))
    counts1 = Counter(labels1)
    counts2 = Counter(labels2)
    n_senses = list(set(labels_all))
    t1 = []
    t2 = []
    label_list = []
    for i in n_senses:
        if counts1[i] + counts2[i] > threshold:
            t1.append(counts1[i])
            t2.append(counts2[i])
            label_list.append(i)
    t1 = np.array(t1)
    t2 = np.array(t2)

    emb1_means = np.array([np.mean(embeds1[labels1 == clust], 0) for clust in label_list])
    emb2_means = np.array([np.mean(embeds2[labels2 == clust], 0) for clust in label_list])
    M = np.nan_to_num(np.array([cdist(emb1_means, emb2_means, metric='cosine')])[0], nan=1)
    t1_dist = t1 / t1.sum()
    t2_dist = t2 / t2.sum()
    wass = ot.emd2(t1_dist, t2_dist, M)
    jsd = compute_jsd(t1_dist, t2_dist)
    return jsd, wass


def detect_meaning_gain_and_loss(labels1, labels2, threshold):
    labels1 = list(labels1)
    labels2 = list(labels2)
    all_count = Counter(labels1 + labels2)
    first_count = Counter(labels1)
    second_count = Counter(labels2)
    gained_meaning = False
    lost_meaning = False
    all = 0
    meaning_gain_loss = 0

    for label, c in all_count.items():
        all += c
        if c >= threshold:
            if label not in first_count or first_count[label] <= 2:
                gained_meaning=True
                meaning_gain_loss += c
            if label not in second_count or second_count[label] <= 2:
                lost_meaning=True
                meaning_gain_loss += c
    return str(gained_meaning) + '/' + str(lost_meaning), meaning_gain_loss/all



def compute_divergence_across_many_periods(embeddings, labels, splits, corpus_slices, threshold, method):
    all_clusters = []
    all_embeddings = []
    clusters_dict = {}
    for split_num, split in enumerate(splits):
        # print('split_num: ' + str(split_num))
        # print('split: ' + str(split))
        if split_num > 0:
            clusters = labels[splits[split_num-1]:split]
            clusters_dict[corpus_slices[split_num - 1]] = clusters
            all_clusters.append(clusters)
            # print('cluster of length ' + str(len(clusters)) + ' added')
            ts_embeds = embeddings[splits[split_num - 1]:split]
            all_embeddings.append(ts_embeds)
    all_measures = []
    all_meanings = []
    # print('cluster count', len(all_clusters))
    for i in range(len(all_clusters)):
        if i < len(all_clusters) - 1:
            try:
                # jsd and wass keep returning really weird stuff
                jsd, wass = compute_divergence_from_cluster_labels(all_embeddings[i], all_embeddings[i+1], all_clusters[i], all_clusters[i+1], threshold)
            except:
                # FLAG FLAG FLAG FLAG FLAG for some reason this is giving me issues
                jsd, wass = 0, 0
            meaning, meaning_score = detect_meaning_gain_and_loss(all_clusters[i], all_clusters[i+1], threshold)
            all_meanings.append(meaning)
            if method == 'WD':
                # print('WD identified as method')
                measure = wass
            else:
                measure = jsd
            all_measures.append(measure)
            # print('measure ' + str(measure) + ' appended')
            
    # i think this "entire" jsd/wass stuff is meant for if you have more than one shift in corpus slice.
    # so the repetitions in measure, i think, should be natural—because you have a measure, the average measure,
    # and then the entire measure.
    # the issue is that for all words, the measure is exactly the same.
    try:
        entire_jsd, entire_wass = compute_divergence_from_cluster_labels(all_embeddings[0], all_embeddings[-1], all_clusters[0], all_clusters[-1], threshold)
    except:
        entire_jsd, entire_wass = 0, 0
    meaning, meaning_score = detect_meaning_gain_and_loss(all_clusters[0],all_clusters[-1], threshold)
    all_meanings.append(meaning)


    avg_measure = sum(all_measures)/len(all_measures)
    try:
        measure = entire_wass
    except:
        measure = 0
    all_measures.extend([measure, avg_measure])
    all_measures = [float("{:.6f}".format(score)) for score in all_measures]
    return all_measures, all_meanings, clusters_dict



def main():
    """Script for running clustering method"""
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
    parser.add_option('--clustering-method', 
                      type=str, 
                      default='WD',
                      help='Method used for clustering: default=%default; otherwise JSD is used')
    parser.add_option('--merging-threshold',
                      type=int,
                      default=0,
                      help='First id of terms to process: default=%default')
    parser.add_option('--epsilon',
                      type=float,
                      default=0.10,
                      help='Cosine distance between embeddings threshold: default=%default')
    parser.add_option('--terms-start-id',
                      type=int,
                      default=0,
                      help='First id of terms to process: default=%default')
    parser.add_option('--terms-end-id',
                      type=int,
                      default=-1,
                      help='Last id of terms (inclusive) to process: default=%default')
    parser.add_option('--target-words-only', 
                      action='store_true',
                      help='Whether to do clustering for target words only')
    parser.add_option('--additional-info', 
                      action='store_true',
                      help='Whether to record additional clustering info')
    parser.add_option('--seed',
                      type=int,
                      default=123,
                      help='Random seed: default=%default')
    

    (options, args) = parser.parse_args()
    # print(args)

    dataset = options.dataset
    model_name = options.model
    method = options.clustering_method
    threshold = options.merging_threshold
    epsilon = options.epsilon
    terms_start_id = options.terms_start_id
    terms_end_id = options.terms_end_id
    target_words_only = options.target_words_only
    get_additional_info = options.additional_info
    seed = options.seed

    
    if dataset == 'semeval_en':
        period_1_str = 'corpus1'
        period_2_str = 'corpus2'
    elif dataset == 'LiverpoolFC':
        period_1_str = 'period_2011-13'
        period_2_str = 'period_2017'
    else:
        period_1_str = 'period_1'
        period_2_str = 'period_2'
    
    corpus_slices = ['period_1', 'period_2']
    
    datadir = '../data/{}/processed_{}'.format(dataset, extract_model_name_from_path(model_name))
    embeddings_dir = '../representations/{}__{}'.format(dataset.lower(), extract_model_name_from_path(model_name))
    
    if target_words_only:
        results_path = '../representations/{}__{}/montariol_results_for_targets'.format(dataset.lower(), extract_model_name_from_path(model_name))    
    else:
        results_path = '../representations/{}__{}/montariol_results'.format(dataset.lower(), extract_model_name_from_path(model_name))    


    measure_vec = []
    cosine_dist_vec = []
    sentence_dict = {}
    aff_prop_labels_dict = {}
    aff_prop_centroids_dict = {}
    kmeans_5_labels_dict = {}
    kmeans_5_centroids_dict = {}
    kmeans_7_labels_dict = {}
    kmeans_7_centroids_dict = {}
    
    
    if not os.path.exists(results_path):
        os.makedirs(results_path)
        
    
    with open('../data/{}/targets.json'.format(dataset)) as f:
        targets_to_true_score = json.load(f)
    
    with open(os.path.join(datadir, 'target_indices.json')) as f:
        target_indices = json.load(f)
    
    with open(os.path.join(datadir, 'control_indices.json')) as f:
        control_indices = json.load(f)
    
    with open(os.path.join(datadir, 'controls.json')) as f:
        control_terms = json.load(f)
    control_terms = list(control_indices.keys())



    # Get sentence IDs per term per period

    term_sent_ids_by_period = {}
    for term in target_indices:
        term_sent_ids_by_period[term] = {}
        for line_id, period, _, _ in target_indices[term]:
            if period not in term_sent_ids_by_period[term]:
                term_sent_ids_by_period[term][period] = []
            term_sent_ids_by_period[term][period].append(line_id)

    if not target_words_only:
        for term in control_indices:
            term_sent_ids_by_period[term] = {}
            for line_id, period, _, _ in control_indices[term]:
                if period not in term_sent_ids_by_period[term]:
                    term_sent_ids_by_period[term][period] = []
                term_sent_ids_by_period[term][period].append(line_id)

    # Separate a slice of terms to iterate over according to the 
    
    terms_keys = sorted(list(term_sent_ids_by_period.keys()))
    print("# of terms in the vocabulary:", len(terms_keys))
    max_index = min(terms_end_id, len(terms_keys)-1)
    print("selected indices: start_idx={}, end_idx={}".format(terms_start_id, max_index))
    
    select_terms = terms_keys[terms_start_id:max_index] + [terms_keys[max_index]]

    # Get embeddings per term per period
    
    all_embeddings_dict = {}
    for term in tqdm(select_terms, desc='Prepare embeddings'):
        if len(term_sent_ids_by_period[term].keys()) < 2:
            continue

        if term in targets_to_true_score:
            word_emb_dir = os.path.join(embeddings_dir, 'target_embeddings')
        else:
            word_emb_dir = os.path.join(embeddings_dir, 'control_embeddings')

        if not os.path.isfile(os.path.join(word_emb_dir, term+'_embeddings.pickle')):
            print('ERROR:', term)
            continue
        with open(os.path.join(word_emb_dir, term+'_embeddings.pickle'), 'rb') as f:
            embeddings = pickle.load(f)
    
        # accumulate embeddings into a tuples list, where the first tuple element is the sum of embeddings e_m
        # and the second tuple element is the number of embeddings summed up in e_m
        period_1_L_embs_list = []
        for line_id in term_sent_ids_by_period[term][period_1_str]:
            if line_id in embeddings:
                if type(embeddings[line_id]) == list:
                    e_new = embeddings[line_id][0]
                else: 
                    e_new = embeddings[line_id]
    
                if len(period_1_L_embs_list) == 0:
                    period_1_L_embs_list.append((e_new, 1))
                    continue
                    
                under200 = len(period_1_L_embs_list) < 200
                smallest_dist = 1
                smallest_dist_emb_id = 0
                for i in range(len(period_1_L_embs_list)):
                    e_m, _ = period_1_L_embs_list[i]
                    dist_to_e_m = cosine(e_m, e_new) 
                    if dist_to_e_m < smallest_dist:
                        smallest_dist = dist_to_e_m
                        smallest_dist_emb_id = i
                if under200 and smallest_dist > epsilon:
                    period_1_L_embs_list.append((e_new, 1))
                else:
                    e_m, embs_count = period_1_L_embs_list[smallest_dist_emb_id]
                    period_1_L_embs_list[smallest_dist_emb_id] = (e_m + e_new, embs_count+1)
                    
        period_2_L_embs_list = []
        for line_id in term_sent_ids_by_period[term][period_2_str]:
            if line_id in embeddings:
                if type(embeddings[line_id]) == list:
                    e_new = embeddings[line_id][0]
                else: 
                    e_new = embeddings[line_id]
    
                if len(period_2_L_embs_list) == 0:
                    period_2_L_embs_list.append((e_new, 1))
                    continue
                    
                under200 = len(period_2_L_embs_list) < 200
                smallest_dist = 1
                smallest_dist_emb_id = 0
                for i in range(len(period_2_L_embs_list)):
                    e_m, _ = period_2_L_embs_list[i]
                    dist_to_e_m = cosine(e_m, e_new) 
                    if dist_to_e_m < smallest_dist:
                        smallest_dist = dist_to_e_m
                        smallest_dist_emb_id = i
                if under200 and smallest_dist > epsilon:
                    period_2_L_embs_list.append((e_new, 1))
                else:
                    e_m, embs_count = period_2_L_embs_list[smallest_dist_emb_id]
                    period_2_L_embs_list[smallest_dist_emb_id] = (e_m + e_new, embs_count+1)
    
        all_embeddings_dict[term] = {'period_1': [e_m/embs_count for e_m, embs_count in period_1_L_embs_list],
                                     'period_2': [e_m/embs_count for e_m, embs_count in period_2_L_embs_list]}
        

    # Do clustering
    
    results = []
    
    columns = ['word']
    extract_methods = ['AP', 'K5', 'K7', 'FREQ', 'MEANING GAIN/LOSS']
    for extract_method in extract_methods:
        for num_slice, cs in enumerate(corpus_slices):
            if extract_method == 'FREQ':
                columns.append(extract_method + ' ' + cs)
            else:
                if num_slice < len(corpus_slices) - 1:
                    columns.append(extract_method + ' ' + cs + '-' + corpus_slices[num_slice + 1])
        columns.append(extract_method + ' All')
        if extract_method != 'MEANING GAIN/LOSS':
            columns.append(extract_method + ' Avg')
    
    for i, word in tqdm(list(enumerate(select_terms)), desc="Run clustering"):
        
        # if os.path.isfile(os.path.join(results_path, '{}__results_dct.json'.format(word))):
        #     with open(os.path.join(results_path, '{}__results_dct.json'.format(word))) as f:
        #         dct = json.load(f)
        #     if len(dct) > 0:
        #         continue
        
        if word not in all_embeddings_dict:
            continue
    
        try:
            emb = all_embeddings_dict[word]

            
            all_embeddings = []
            all_sentences = {}
            splits = [0]
            all_slices_present = True
            all_freqs = []
    
            # combined embeddings from each period
            for cs in corpus_slices:
                cs_embeddings = []
                cs_sentences = []
    
                count_all = 0
                text_seen = set()
                
                all_freqs.append(len(emb[cs]))
                cs_text = cs + '_text'
                # print("Slice: ", cs, "Num embeds: ", len(emb[cs]))
                num_sent_codes = 0
                
                for idx in range(len(emb[cs])):
                    # try:
                    #     # for a special format of the embeddings dict
                    #     e, count_emb = emb[cs][idx]
                    #     e = e/count_emb
                    # except:
                    #     # for standard format of the embeddings dict
                    #     e = emb[cs][idx]
    
                    # working with standard format of the embeddings dict
                    cs_embeddings.append( emb[cs][idx])
    
                all_embeddings.append(np.array(cs_embeddings))
                splits.append(splits[-1] + len(cs_embeddings))
                
            embeddings_concat = np.concatenate(all_embeddings, axis=0)
            
            if embeddings_concat.shape[0] < 7 or not all_slices_present:
                continue
            else:
                kmeans_5_labels, kmeans_5_centroids = cluster_word_embeddings_k_means(embeddings_concat, 5, seed)
                kmeans_5_labels = combine_clusters(kmeans_5_labels, embeddings_concat, threshold, remove=[])
                all_kmeans5_measures, all_meanings, clustered_kmeans_5_labels = compute_divergence_across_many_periods(embeddings_concat, kmeans_5_labels, splits, corpus_slices, threshold, method)
                kmeans_7_labels, kmeans_7_centroids = cluster_word_embeddings_k_means(embeddings_concat, 7, seed)
                kmeans_7_labels = combine_clusters(kmeans_7_labels, embeddings_concat, threshold, remove=[])
                all_kmeans7_measures, all_meanings, clustered_kmeans_7_labels = compute_divergence_across_many_periods(embeddings_concat, kmeans_7_labels, splits, corpus_slices, threshold, method)
    
                aff_prop_labels, aff_prop_centroids = cluster_word_embeddings_aff_prop(embeddings_concat)
                aff_prop_labels = combine_clusters(aff_prop_labels, embeddings_concat, threshold=threshold, remove=[])
                all_aff_prop_measures, all_meanings, clustered_aff_prop_labels = compute_divergence_across_many_periods(embeddings_concat, aff_prop_labels, splits, corpus_slices, threshold, method)
    
                all_freqs = all_freqs + [sum(all_freqs)] + [sum(all_freqs)/len(all_freqs)]
    
                word_results = [word] + all_aff_prop_measures + all_kmeans5_measures + all_kmeans7_measures + all_freqs + all_meanings # need to add back in all_freqs
                # print("Results:", word, word_results)
    
                with open(os.path.join(results_path, '{}__results_dct.json'.format(word)), 'w') as f:
                    json.dump(dict(list(zip(columns, word_results))), f)
    
            results.append(word_results)
    
            
            
            # how to interpret— word, then first 3 are the divergence for kmeans 5, second 3 are kmeans 7, next 3 nums are frequencies, then "meaning" measure then True/False if gained meaning, False/True if lost meaning.
            # Results: ['tapping', 0.092227, 0.092227, 0.092227, 0.100495, 0.100495, 0.100495, 62, 653, 715, 357.5, 'True/False', 'True/False']
            
            if get_additional_info:
                sentence_dict[word] = all_sentences
                aff_prop_labels_dict[word] = clustered_aff_prop_labels
                aff_prop_centroids_dict[word] = aff_prop_centroids
    
                kmeans_5_labels_dict[word] = clustered_kmeans_5_labels
                kmeans_5_centroids_dict[word] = kmeans_5_centroids
    
                kmeans_7_labels_dict[word] = clustered_kmeans_7_labels
                kmeans_7_centroids_dict[word] = kmeans_7_centroids  # add results to dataframe for saving
    
            # processed_words.append(word)
    
        except Exception as err:
            print('ERROR', word)
            print(err)
    
            
    if get_additional_info:
        # save cluster labels and sentences to pickle
        dicts = [(aff_prop_centroids_dict, 'aff_prop_centroids'), (aff_prop_labels_dict, 'aff_prop_labels'),
                 (kmeans_5_centroids_dict, 'kmeans_5_centroids'), (kmeans_5_labels_dict, 'kmeans_5_labels'),
                 (kmeans_7_centroids_dict, 'kmeans_7_centroids'), (kmeans_7_labels_dict, 'kmeans_7_labels'),
                 (sentence_dict, "sents")]
    
        for data, name in dicts:
            data_file = os.path.join(results_path, name + ".pkl")
            pf = open(data_file, 'wb')
            pickle.dump(data, pf)
            pf.close()
    
    



    
if __name__ == "__main__":
    main()