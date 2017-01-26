import numpy
import ast
import sys
import cPickle as pickle
import pprint
import string
import re
import operator


###############################################################################
###                        Global Settings and Paths                        ###
###############################################################################


data_path = '/Users/corbinrosset/Dropbox/Arora/QA-data/VanDurme_FB_annotations/annotated_clueweb/ClueWeb09_English_1/processed/'
execute_path = '/Users/corbinrosset/Dropbox/Arora/QA-code/process_clueweb/'
manifest_file = 'manifest.txt'
WRITE_OUTPUT_TO_FILE = False ### overwrite .processed files which contain raw text with massaged and filtered text. Make this false if the .final files already exist
USE_FINAL = True ### use .final files, for which text has been processed. If this is True, then you probably don't need to do all the tasks in process()


###############################################################################
###                   Data Structures and Things to Count                   ###
###############################################################################

rltn_freq = {} ### count occurances of relationships
enty_id_freq = {} ### count occurances of entity IDs overall
enty_raw_freq = {} ### count occurances of text entities overall, match above?
enty_id_string_map = {} ### map FB ID to entity string
vocab_count = {} ### count occurances of words
rltn_word_freq = {} ### histogram of words associated with each relationship
rltn_input_freq = {} ### histogram of entities incoming to a relationship
rltn_output_freq = {} ### histogram of entities incoming to a relationship
triples_count = {} ### count occurances of triples in clue web data

### This takes a pickled *.processed file (list of lists of text mentions) and 
### generates some statistics and data structures:
'''
* vocabulary and frequencies of relations
* vocabulary of entities and their frequencies
* type of relation - one to one, many to one, many to many
	- for each relationship, compute the avg and std of number of unique head entities and tail entities
* vocabulary of natural language
* histogram of number of words per textual mention
* frequency counts of entity pairs, as well as how often they are connected by 
	a specific relationship

Some needed data structures going forward:
* count of each word associated with each textual mention of a relationship
	- a R by V matrix, R relationships, V words in vocabulary
* an R by N^2 sparse matrix of entity pairs associated with relationships
'''

###############################################################################
###                         Helper Methods                                  ###
###############################################################################


def process_text(text):
	'''return a list of words after removing ugly things from raw string'''

	text = text.lower()
	### first remove hex numbers in text like \xe2 and parentheses
	text = re.sub(r'[^\x20-\x7e]', '', text)
	
	### then remove quotation marks and punctuation
	text = re.sub(r'["!,\\&#()+<>=?\[\]@\']','', text)

	### replace '-'' with ' '
	text = re.sub(r'[-;:/]',' ', text)

	### split on commas and white space
	text = re.split('; |, |\s+', text.strip())
	# text = map(lambda x: x.lower(), line[3].strip().split())
	text = filter(lambda x: x is not None, [process_word(x) for x in text])
	return text

def process_word(word):

	### preemptively continue
	if word == '' or word in exclude or len(word) == 1:
		return None

	### perhaps try to match URLs??

	### replace all $ occurances with DOLLAR
	word = re.sub(r'\$\d+', 'DOLLAR', word)

	### replace all years with YEAR
	word = re.sub(r'\d{4}s|\d{4}', 'YEAR', word)

	### replace percents with PERCENT
	word = re.sub(r'\d+\.?\d+%', 'PERC', word)

	### replace all other string with numerals as NUMBER
	if is_number(word):
		word = 'NUM'

	if word in dictionary:
		return word 
	else:
		return None
def insert_or_add(table, key, value):
	if key in table:
		table[key].add(value)
	else:
		table[key] = set([value])

def insert_or_increment(table, key):
	if key in table:
		table[key] += 1
	else:
		table[key] = 1

def insert_or_append(table, key, value):
	if key in table:
		table[key].append(value)
	else:
		table[key] = [value]

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False	

###############################################################################
###                         Primary Method                                  ###
###############################################################################

def process(formatted_data_chunk, name):
	global rltn_freq
	global enty_id_freq
	global enty_raw_freq
	global enty_id_string_map
	global vocab_count
	global rltn_word_freq
	global rltn_input_freq
	global rltn_output_freq

	if WRITE_OUTPUT_TO_FILE:
		name = name.replace('processed', 'final')
		new_output = open(data_path + name, 'w')
	last_str = ''
	for i, line in enumerate(formatted_data_chunk):
		### [relationships, reversd, (leftEntity, leftEntityID), result, (rightEntity, rightEntityID)]
		line = ast.literal_eval(line) ### convert text to data structure
		reversd = int(line[1])
		relationships = line[0]
		left = line[2]
		text = line[3] 
		right = line[4]
		
		### MAJOR PROBLEM: MANY MIDs ARE CLASHING, MULTIPLE DEFINITIONS
		### map entity names to ids and enforce consistency
		### if left[0] in enty_id_string_map:
		### 	if enty_id_string_map[left[0]] != left[1]:
		### 		# print 'ERROR: ' + line
		### 		print 'ERROR: ' + str(left[0]) + ' associated with ' + str(left[1]) + ' but is already mapped to ' + str(enty_id_string_map[left[0]])
		### 		continue
		### else:
		### 	enty_id_string_map[left[0]] = left[1]
		### if right[0] in enty_id_string_map:
		### 	if enty_id_string_map[right[0]] != right[1]:
		### 		# print 'ERROR: ' + line
		### 		print 'ERROR: ' + str(right[0]) + ' associated with ' + str(right[1]) + ' but is already mapped to ' + str(enty_id_string_map[right[0]])
		### 		continue
		### else:
		### 	enty_id_string_map[right[0]] = right[1]]
		

		# ### update the entity string to ID  number map
		# insert_or_add(enty_id_string_map, right[0], right[1])


		# ### count relationships
		# for rltn in relationships:
		# 	insert_or_increment(rltn_freq, rltn)
		# 	if rltn not in rltn_word_freq:
		# 		rltn_word_freq[rltn] = {}
		# 	if rltn not in rltn_input_freq:
		# 		rltn_input_freq[rltn] = {}
		# 	if rltn not in rltn_output_freq:
		# 		rltn_output_freq[rltn] = {}

		# ### count entities
		# insert_or_increment(enty_raw_freq, left[0])
		# ### insert_or_increment(enty_id_freq, left[1]) ### dont trust MIDs
		# insert_or_increment(enty_raw_freq, right[0])
		# ### insert_or_increment(enty_id_freq, right[1]) ### dont trust MIDs

		processed_text = text.strip().split()
		if not USE_FINAL:
			processed_text = process_text(text)

		### count triples only if they have valid text
		if len(processed_text) > 0:
			if reversd == 1:
				insert_or_increment(triples_count, (right, tuple(relationships), left))
			else:
				insert_or_increment(triples_count, (left, tuple(relationships), right))
		
		# ### count vocabulary words
		# for word in processed_text:
		# 	if word == None:
		# 		continue
		# 	### do processing of word finally
		# 	insert_or_increment(vocab_count, word)
		# 	for rltn in relationships:
		# 		insert_or_increment(rltn_word_freq[rltn], word)

		# ### count in-degrees and out-degree of each relationship wrt entities
		# if reversd == 0:
		# 	insert_or_increment(rltn_input_freq[rltn], left[0])
		# 	insert_or_increment(rltn_output_freq[rltn], right[0])
		# else:
		# 	insert_or_increment(rltn_input_freq[rltn], right[0])
		# 	insert_or_increment(rltn_output_freq[rltn], left[0])

		if WRITE_OUTPUT_TO_FILE:
			line[3] = ' '.join(processed_text)
			new_output.write(str(line) + '\n')
	if WRITE_OUTPUT_TO_FILE:
		new_output.close()


###############################################################################
###                                 Main                                    ###
###############################################################################


### the format of processed_data_chunk is a list of lists of the form [relationships, boolean reversd, (leftEntity, leftEntityID), parsed relationship string, (rightEntity, rightEntityID)]

### build list of exclusions and stop words
exclude = set(string.punctuation)
exclude.add('--')
exclude.add('``')

### and add to it a list of stop words to remove from dictionary
for stopword in open(execute_path + 'stopwords.txt', 'r'):
	exclude.add(stopword.strip().lower())
print 'loaded ' + str(len(exclude)) + ' stopwords'

### build dictionary of valid english words
dictionary = set([])
### to represent common numbers and values, convert them to:
dictionary.add('DOLLAR')
dictionary.add('YEAR')
dictionary.add('PERC')
dictionary.add('NUM') ### contains '#' or not one of the above

for dictionary_word in open(execute_path + 'dictionary.txt', 'r'):
	if dictionary_word not in exclude:
		dictionary.add(dictionary_word.strip().lower())
print 'loaded dictionary of ' + str(len(dictionary)) + ' words'

### open manifest of formatted data
try:
	manifest_list = open(data_path + manifest_file, 'r')
	print 'loaded manifest of which data files need to be processed...'
except:
	print 'no manifest file, aborting'
	exit()

### collect statistics of the corpus chunk by chunk
for chunk in manifest_list:
	chunk = chunk.strip()
	if USE_FINAL: 
		chunk = chunk.replace('processed', 'final') ### use only final files
	try:
		formatted_data_chunk = open(data_path + chunk, 'r')
		print 'processing ' + str(chunk)
	except:
		print 'no \".processed\" file exists, run parse_text_anchors_vandurme.py on a corpus'
		continue
	process(formatted_data_chunk, chunk)


###############################################################################
###                         Persist Output                                  ###
###############################################################################


### summarize all results
print 'done computing statistics...outputting files...'

report = open(data_path + 'triple_count.txt', 'w')
report.write('number of unique triples: ' + str(len(triples_count)) + '\n')
report.write('\n======================================================\n')
report.write(str(pprint.pformat(sorted(triples_count.items(), key=operator.itemgetter(1), reverse=True))))
report.close()

# report = open(data_path + 'relation_freq.txt', 'w')
# report.write('number of unique relationships: ' + str(len(rltn_freq)) + '\n')
# report.write('\n======================================================\n')
# report.write(str(pprint.pformat(sorted(rltn_freq.items(), key=operator.itemgetter(1), reverse=True))))
# report.close()

# report = open(data_path + 'entity_id_freq.txt', 'w')
# report.write('number of unique entity IDs (MIDs): ' + str(len(enty_id_freq)) + '\n')
# report.write('\n======================================================\n')
# report.write(str(pprint.pformat(sorted(enty_id_freq.items(), key=operator.itemgetter(1), reverse=True))))
# report.close()

# report = open(data_path + 'entity_plaintext_freq.txt', 'w')
# report.write('number of unique entity strings: ' + str(len(enty_raw_freq)) + '\n')
# report.write('\n======================================================\n')
# report.write(str(pprint.pformat(sorted(enty_raw_freq.items(), key=operator.itemgetter(1), reverse=True))))
# report.close()

# report = open(data_path + 'entity_id_plaintext_map.txt', 'w')
# report.write('number of unique entity strings: ' + str(len(enty_id_string_map)) + '\n')
# report.write('\n======================================================\n')
# for enty in enty_id_string_map:
# 	if len(enty_id_string_map[enty]) > 1:
# 		report.write(str(enty) + ' has many mappings: ' + str(enty_id_string_map[enty]) + '\n')
# report.write('\n======================================================\n')
# report.write(str(pprint.pformat(enty_id_string_map)))
# report.close()

# report = open(data_path + 'vocab_count.txt', 'w')
# report.write('vocab size: ' + str(len(vocab_count)) + '\n')
# report.write('\n======================================================\n')
# report.write(str(pprint.pformat(sorted(vocab_count.items(), key=operator.itemgetter(1), reverse=True))))
# report.close()

# report = open(data_path + 'relation_word_counts.txt', 'w')
# unique_words_per_relation = {}
# for rltn in rltn_word_freq:
# 	unique_words_per_relation[rltn] = len(rltn_word_freq[rltn])
# report.write('number of unique words associated with each relation:\n')
# report.write(str(pprint.pformat(sorted(unique_words_per_relation.items(), key=operator.itemgetter(1), reverse=True))))
# report.write('\n======================================================\n')
# report.write(str(pprint.pformat(sorted(rltn_word_freq.items(), key=operator.itemgetter(1), reverse=True))))
# report.close()

# report = open(data_path + 'relation_input_entity_freq.txt', 'w')
# unique_inputs_per_relation = {}
# for rltn in rltn_input_freq:
# 	unique_inputs_per_relation[rltn] = len(rltn_input_freq[rltn])
# report.write('number of unique input entities for each relation:\n')
# report.write(str(pprint.pformat(sorted(unique_inputs_per_relation.items(), key=operator.itemgetter(1), reverse=True))))
# # for rltn in rltn_input_freq:
# # 	report.write('number of unique input entities for ' + str(rltn) + ': ' + str(len(rltn_input_freq[rltn])) + '\n')
# report.write('\n======================================================\n')
# report.write(str(pprint.pformat(sorted(rltn_input_freq.items(), key=operator.itemgetter(1), reverse=True))))
# report.close()

# report = open(data_path + 'relation_output_entity_freq.txt', 'w')
# unique_output_per_relation = {}
# for rltn in rltn_input_freq:
# 	unique_output_per_relation[rltn] = len(rltn_output_freq[rltn])
# report.write('number of unique output entities for each relation:\n')
# report.write(str(pprint.pformat(sorted(unique_output_per_relation.items(), key=operator.itemgetter(1), reverse=True))))
# # for rltn in rltn_output_freq:
# # 	report.write('number of unique output entities for ' + str(rltn) + ': ' + str(len(rltn_output_freq[rltn])) + '\n')
# report.write('\n======================================================\n')
# report.write(str(pprint.pformat(sorted(rltn_output_freq.items(), key=operator.itemgetter(1), reverse=True))))
# report.close()

###############################################################################
###                         Pickle the Tables                               ###
###############################################################################

### print sizes:
print 'data structure sizes in bytes:\n'
print 'triples_count: ' + str(sys.getsizeof(triples_count))
print 'rltn_freq: ' + str(sys.getsizeof(rltn_freq))
print 'enty_id_freq: ' + str(sys.getsizeof(enty_id_freq))
print 'enty_raw_freq: ' + str(sys.getsizeof(enty_raw_freq))
print 'enty_id_string_map: ' + str(sys.getsizeof(enty_id_string_map))
print 'vocab_count: ' + str(sys.getsizeof(vocab_count))
print 'rltn_word_freq: ' + str(sys.getsizeof(rltn_word_freq))
print 'rltn_input_freq: ' + str(sys.getsizeof(rltn_input_freq))
print 'rltn_output_freq: ' + str(sys.getsizeof(rltn_output_freq))

### write data structures to persisted files
pickle.dump(triples_count, open(data_path + 'triple_count.pkl', 'w'), pickle.HIGHEST_PROTOCOL)
# pickle.dump(rltn_freq, open(data_path + 'rltn_freq.pkl', 'w'), pickle.HIGHEST_PROTOCOL)
# pickle.dump(enty_id_freq, open(data_path + 'enty_id_freq.pkl', 'w'), pickle.HIGHEST_PROTOCOL)
# pickle.dump(enty_raw_freq, open(data_path + 'enty_raw_freq.pkl', 'w'), pickle.HIGHEST_PROTOCOL)
# pickle.dump(enty_id_string_map, open(data_path + 'enty_id_string_map.pkl', 'w'), pickle.HIGHEST_PROTOCOL)
# pickle.dump(vocab_count, open(data_path + 'vocab_count.pkl', 'w'), pickle.HIGHEST_PROTOCOL)
# pickle.dump(rltn_word_freq, open(data_path + 'rltn_word_freq.pkl', 'w'), pickle.HIGHEST_PROTOCOL)
# pickle.dump(rltn_input_freq, open(data_path + 'rltn_input_freq.pkl', 'w'), pickle.HIGHEST_PROTOCOL)
# pickle.dump(rltn_output_freq, open(data_path + 'rltn_output_freq.pkl', 'w'), pickle.HIGHEST_PROTOCOL)
