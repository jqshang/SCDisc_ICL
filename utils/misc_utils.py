import re
import math

def extract_model_name_from_path(model_path):
	if model_path[-1] == '/':
		return model_path[:-1].split('/')[-1]
	return model_path.split('/')[-1]

def clean_tokenizer_formatting(token, model_name):
	if 'xl-lexeme' in model_name or 'xlm-roberta' in model_name:
		# for SentencePiece tokenizers 
		return re.sub('\u2581', '', token)
	else:
		# for WordPiece tokenizers
		return re.sub('##', '', token)

def is_token_piece(token, model_name):
	if 'xl-lexeme' in model_name or 'xlm-roberta' in model_name:
		# for SentencePiece tokenizers 
		return not token.startswith('\u2581')
	else:
		# for WordPiece tokenizers
		return token.startswith('##')

def combine_tokens_into_clean_text(tokens, model_name):
	if 'xl-lexeme' in model_name or 'xlm-roberta' in model_name:
		# for SentencePiece tokenizers 
		return re.sub('\u2581', ' ', ''.join(tokens)).strip()
	else:
		# for WordPiece tokenizers
		return re.sub(' ##', '', ' '.join(tokens)).strip()

def convert_token_lst_to_tokenizer_format(token_lst, model_name):
	new_token_lst = []
	if 'xl-lexeme' in model_name or 'xlm-roberta' in model_name:
		# for SentencePiece tokenizers 
		new_token_lst.append('\u2581'+token_lst[0])
		if len(token_lst) > 1:
			for tk in token_lst[1:]:
				new_token_lst.append(tk)
		return new_token_lst
	else:
		# for WordPiece tokenizers
		new_token_lst.append(token_lst[0])
		if len(token_lst) > 1:
			for tk in token_lst[1:]:
				new_token_lst.append('##'+tk)
		return new_token_lst

def count_combinations(n, r):
    try: 
        return math.factorial(n) / (math.factorial(r) * math.factorial(n - r))
    except OverflowError:
        return 1e6

def compute_p_value(val, distribution):
    return sum([1 for v in distribution if v >= val])/len(distribution)