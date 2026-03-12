import os
import re
import json
import numpy as np
from optparse import OptionParser
from tqdm import tqdm
from collections import Counter
from misc_utils import *


# Align the lemmatized and original corpora from SemEval

def main():
    usage = "%prog "
    parser = OptionParser(usage=usage)
    parser.add_option('--dataset',
                      type=str,
                      default='semeval_en',
                      help='Dataset directory name: default:%default')
    parser.add_option('--tokenizer-model',
                      type=str,
                      default='bert-base-uncased',
                      help='Tokenizer model to use: default=%default')
    parser.add_option('--lemmas-file',
                      type=str,
                      default='',
                      help='Input file with prelemmatized sentences. Formatted as follows:\n\
                                {"id": "ccoha_000000004", \
                                "lemmas": ["distance",..., "answer"]", \
                                "source": "corpus1"}')

    (options, args) = parser.parse_args()

    dataset = options.dataset
    tokenizer_model = options.tokenizer_model
    lemmas_file = options.lemmas_file

    outdir = os.path.join('../data', dataset, 'processed_' + extract_model_name_from_path(tokenizer_model))
    
    with open(os.path.join(outdir, 'tokenized_all.jsonlist')) as f:
        tokenized_data = [json.loads(line) for line in f.readlines()]


    if lemmas_file == "":
        infile = os.path.join(outdir, 'lemmatized_all.jsonlist') 
    else:
        infile = lemmas_file
    
    with open(infile) as f:
        lemmatized_data = [json.loads(line) for line in f.readlines()]

    assert len(tokenized_data) == len(lemmatized_data)
    
    alignment_information = []
    for i in tqdm(range(len(tokenized_data))):
        tokens_line_id = tokenized_data[i]['id']
        lemmas_line_id = lemmatized_data[i]['id']
        assert tokens_line_id == lemmas_line_id
        tokens = [''.join(lst) for lst in tokenized_data[i]['tokens']]
        lemmas = lemmatized_data[i]['lemmas']

        tokens_clean, tokens_removed = remove_punctuation(tokens)
        lemmas_clean, lemmas_removed = remove_punctuation(lemmas)

        alignment = align_lists(tokens_clean, lemmas_clean)

        corrected_alignment = fix_alignment(tokens_removed, lemmas_removed, alignment)

        # score the mapping
        try:
            score = score_alignment(tokens, lemmas, corrected_alignment)
        except IndexError as e:
            print("Index error:", e)
            print(line_i)
            print(tokens)
            print(lemmas)
            print(tokens_clean)
            print(lemmas_clean)
            print(alignment)
            print(corrected_alignment)
            raise e
        word_match_score = score_word_matching(tokens, lemmas, corrected_alignment)
        # tt_alignment)
    
        alignment_information.append({'id': tokens_line_id, 'score': score, 'word_match_score': word_match_score, 'alignment': sorted(corrected_alignment)})

    # df = pd.DataFrame()
    # df['id'] = line_ids
    # df['tokens'] = tokenized_strings
    # df['lemmas'] = lemmatized_strings
    # df['score'] = scores
    # df['word_match_score'] = word_match_scores
    # df['alignment'] = alignments
    
    # df.to_csv(os.path.join(outdir, 'matching.csv'))

    if lemmas_file != "":
        outfile = 'pre_lemmas_matching.jsonlist'
    else:
        outfile = 'lemmas_matching.jsonlist'
    with open(os.path.join(outdir, outfile), 'w') as f:
        for line in alignment_information:
            f.write(json.dumps(line) + '\n')




def score_word_matching(list_one, list_two, mapping):
    # Score the alignment of two lists, using edit distance
    score = 0
    for i, pair in enumerate(mapping):
        index1, index2 = pair
        if index1 < 0:
            score += 1
        elif index2 < 0:
            score += 1
        elif list_one[index1] is None or list_two[index2] is None:
            score += 1
        elif list_one[index1] != list_two[index2]:
            score += 1
    return score


def score_alignment(list_one, list_two, mapping, debug=False, ignore_chars=',.;?!:()"'):
    # Score the alignment of two lists, using edit distance
    score = 0
    for i, pair in enumerate(mapping):
        index1, index2 = pair
        if index1 < 0:
            score += len(list_two[index2])
            if debug:
                print('---', list_two[index2], len(list_two[index2]))
        elif index2 < 0:
            score += len(list_one[index1])
            if debug:
                print(list_one[index1], '---', len(list_one[index1]))
        elif list_one[index1] is None:
            if list_two[index2] in ignore_chars:                
                dist = 0
            else:
                dist = len(list_two[index2])
            score += dist
            if debug:
                print('---', list_two[index2], score)
        elif list_two[index2] is None:
            if list_one[index1] in ignore_chars:
                dist = 0
            else:
                dist = len(list_one[index1])
            score += dist
            if debug:
                print(list_one[index1], '---', dist)
        elif list_one[index1] != list_two[index2]:
            dist = levenshteinDistance(list_one[index1], list_two[index2])+1
            if debug:
                print(list_one[index1], list_two[index2], dist)
            score += dist
        elif debug:
            print(list_one[index1], list_two[index2], 0)
    return score
        
        
def levenshteinDistance(s1, s2):
    # Compute the edit distance between the strings

    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    max_dist = max(len(s1), len(s2))
    return min(max_dist, distances[-1])


def remove_punctuation(tokens, punct=',.;?!:()"'):
    # remove punctuation from a list of tokens
    removed = {}
    for t_i, t in enumerate(tokens):
        if t in punct:
            removed[t_i] = t
    clean = [t for t_i, t in enumerate(tokens) if t_i not in removed]
    return clean, removed


def fix_alignment(tokens_removed, lemmas_removed, alignment):

    for t_i, token in tokens_removed.items():
        corrected = []
        for pair in alignment:            
            if pair[0] < t_i:
                corrected.append(pair)
            else:
                corrected.append((pair[0]+1, pair[1]))
        alignment = corrected[:]        
    alignment.extend([(t_i, -1) for t_i, token in tokens_removed.items()])
    alignment = sorted(alignment)
                
    for t_i, lemma in lemmas_removed.items():
        corrected = []
        for pair in alignment:
            if pair[1] < t_i:
                corrected.append(pair)
            else:
                corrected.append((pair[0], pair[1]+1))
        alignment = corrected[:]
    alignment.extend([(-1, t_i) for t_i, lemma in lemmas_removed.items()])
    alignment = sorted(alignment)
        
    return alignment
    

def align_sublists_on_exact_matches(tokens, lemmas, start_token=0, end_token=None, start_lemma=0, end_lemma=None):
    """Compare subsets of two lists and match the common unique tokens"""

    local_alignment = []
    
    if end_token is None:
        end_token = len(tokens)
    if end_lemma is None:
        end_lemma = len(lemmas)

    # find the tokens that are common to both, but only occur once in each
    unique_tokens = set([t for t, c in Counter(tokens[start_token:end_token]).items() if c == 1])
    unique_lemmas = set([t for t, c in Counter(lemmas[start_lemma:end_lemma]).items() if c == 1])
    common_unique = unique_tokens.intersection(unique_lemmas)
    
    shared_token_indices = []
    for i, token in enumerate(tokens[start_token:end_token]):
        if token in common_unique:
            shared_token_indices.append((start_token+i, token))

    shared_lemma_indices = []
    for i, lemma in enumerate(lemmas[start_lemma:end_lemma]):
        if lemma in common_unique:
            shared_lemma_indices.append((start_lemma+i, lemma))

    lemma_offset = 0
    for i, pair, in enumerate(shared_token_indices):
        token_index, token = pair            
        # make sure there are still more lemmas to compare against
        if i + lemma_offset < len(shared_lemma_indices):
            lemma_index, lemma = shared_lemma_indices[i+lemma_offset]
            # if the token does not match the lemma, consider skipping
            if token != lemma:
                # if there are enough lemmas, check if the token matches the next one
                if i+lemma_offset+1 < len(shared_lemma_indices) and token == shared_lemma_indices[i+lemma_offset+1][1]:
                    # if so, increase the lemma offset and use the next lemma
                    lemma_offset += 1
                    lemma_index, lemma = shared_lemma_indices[i+lemma_offset]
                # alternatively, if there are enough tokens, check if the lemma matches the next one
                elif i+1 < len(shared_token_indices) and lemma == shared_token_indices[i+1][1] and i+lemma_offset+1 > 0:
                    # if so, skip this pair, and decrease the lemma offset
                    lemma_offset -= 1
            # if the token now matches the lemma (if we added to lemma offset), add the alignment pair
            if token == lemma:
                local_alignment.append((token_index, lemma_index))

    return local_alignment



def align_sublists_on_partial_matches(long_list, short_list, long_start=0, long_end=None, short_start=0, short_end=None):
    """
    Compare subsets of two lists and match the common unique tokens
    Assume len(long_list) >= len(short_list)
    """
    
    if long_end is None:
        long_end = len(long_list)
    if short_end is None:
        short_end = len(short_list)
    
    n_long = long_end - long_start
    n_short = short_end - short_start
    
    if n_long == n_short:
        alignment = [(long_start+offset, short_start+offset) for offset in range(n_long)]
        return alignment
    else:
        short_copy = short_list[:]

        # Add Null tokens into matching one by one until there is no more point in doing so
        for iteration in range(n_long-n_short):
            #print("Iteration", iteration)
            scores = []
            # consider adding the null token in each position in the short list
            for null_offset in range(short_end - short_start + 1):
                past_null_offset = 0                
                local_alignment = []
                for token_offset in range(long_end-long_start):
                    #print(token_offset, past_null_offset, token_offset == null_offset, short_start+token_offset-past_null_offset >= short_end, long_start+token_offset, short_start+token_offset-past_null_offset, len(short_copy))
                    if token_offset >= null_offset:
                        past_null_offset = 1
                    if token_offset == null_offset:
                        local_alignment.append((long_start+token_offset, -1))
                    elif short_start+token_offset-past_null_offset >= short_end:
                        local_alignment.append((long_start+token_offset, -1))
                    elif short_copy[short_start + token_offset - past_null_offset] is None:
                        local_alignment.append((long_start+token_offset, -1))
                    else:
                        local_alignment.append((long_start+token_offset, short_start+token_offset-past_null_offset))
                score = score_alignment(long_list, short_copy, local_alignment)

                scores.append(score)
            best_position = np.argmin(scores)
            short_copy.insert(short_start+best_position, None)
            short_end += 1            

        # convert back to original indexing of short
        corrected_alignment = []
        offset = 0
        for token_offset, long_token in enumerate(long_list[long_start:long_end]):
            short_index = short_start + token_offset
            short_token = short_copy[short_index]
            if short_token is None:
                offset += 1
                corrected_alignment.append((long_start+token_offset, -1))
            else:
                corrected_alignment.append((long_start+token_offset, short_start+token_offset-offset))

        return corrected_alignment


            
def align_lists(tokens, lemmas):
    done = False
    n_tokens = len(tokens)
    n_lemmas = len(lemmas)
    alignment = [(-1, -1), (n_tokens, n_lemmas)]
    iteration = 0
    while not done:
        new_alignments = []
        for pair_i, aligned_pair in enumerate(alignment[:-1]):
            token_index, lemma_index = aligned_pair
            next_pair = alignment[pair_i+1]
            next_token_index, next_lemma_index = next_pair        
            if next_token_index - token_index > 1 and next_lemma_index - lemma_index > 1:
                local_alignments = align_sublists_on_exact_matches(tokens, lemmas, token_index+1, next_token_index, lemma_index+1, next_lemma_index)
                new_alignments.extend(local_alignments)
        if len(new_alignments) == 0:
            done = True
        else:
            alignment = sorted(alignment + new_alignments)
            iteration += 1
        if iteration > 20:
            raise RuntimeError("Exceeded 20 iterations")
            
    # Now go through and do alignment of sublists on imperfect matches
    for pair_i, aligned_pair in enumerate(alignment[:-1]):
        token_index, lemma_index = aligned_pair
        next_pair = alignment[pair_i+1]
        next_token_index, next_lemma_index = next_pair        
        if next_token_index - token_index > 1 or next_lemma_index - lemma_index > 1:
            if next_token_index - token_index >= next_lemma_index - lemma_index:                
                local_alignments = align_sublists_on_partial_matches(tokens, lemmas, token_index+1, next_token_index, lemma_index+1, next_lemma_index)
            else:
                local_alignments = align_sublists_on_partial_matches(lemmas, tokens, lemma_index+1, next_lemma_index,  token_index+1, next_token_index)
                local_alignments = [(pair[1], pair[0]) for pair in local_alignments]

            new_alignments.extend(local_alignments)
    alignment = sorted(alignment + new_alignments)

    return alignment[1:-1]
    
    

if __name__ == '__main__':
    main()