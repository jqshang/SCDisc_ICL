liverpoolfc__word_exclusions = [# proper nouns that didn't get filtered out
								'akinfenwa', 'vince', 'troy', 'kouyate', 'mulumbu', 
								'slovenia', 'rudy', 'neymars', 'golo', 'shea', 'mona',
								'marcos', 'emery', 'miereles', 'tite', 'lego', 'martins', 
								'martial', 'morty', 'alexander', 'lacazette', 'navi', 'theresa',
								'barnet', 'geoff', 'dominic', 'trump', 'honda', 'dom', 'ukranian',
								'mcgregor', 'barcareal', 'ings', 'mansfield', 'parker', 'parkers', 'gj',  
								'cans', 'firms', 'virtue', 'bent', 'stones', 'woods', 'samba', 'viking',
								'vans',  # proper name from tragets
								'darkness', 'informs', # frequency filter: too few sents with the dominant pos tag
								'91', '6000', '343', '53', '430am', #numericals
								'broker', 'mp4', #used as part of the url
								'2c', 'er', 'playlist', 'l', 'com', # often misformatted in urls & metadata
								'—', '“', '”', # not words
								'kkk', 'cocks', 'anal', 'pussies',# offensive + potential profanity
								]

semeval_en__word_exclusions =  [# proper nouns that didn't get filtered out
								'nate', 'erin', 'fleetwood', 'marge', 'nate', 'tate', 'mm',
								'vera', 'santee', 'fayette',
								'dune', # frequency filter: too few sents with the dominant pos tag
								'ns', 'gene', 'wan', 'cf', 'ct', 'rd', # tokenization errors						
								]