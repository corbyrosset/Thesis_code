import numpy
import re
# import pickle
import cPickle as pickle
import pprint

### This parses the clueweb09 data annotated with Freebase entity anchors by
### some MT algorithm employed by Van Durme and his student Xuchan.
### This data is private and may not be distributed. 

### an example is (it appears all on one line):
### internet.website.owner	m.05h7w3	m.019rl6	" After his talks with Mr [FREEBASE mid=/m/0ck32l al-Maliki]al-Maliki[/FREEBASE], [FREEBASE mid=/m/09q2t Brown]Brown[/FREEBASE] went on to hold further meetings with the [FREEBASE mid=/m/09c7w0 US]US[/FREEBASE] commander in [FREEBASE mid=/m/0d05q4 Iraq]Iraq[/FREEBASE] General [FREEBASE mid=/m/08_hns David Petraeus]David Petraeus[/FREEBASE] and [FREEBASE mid=/m/0d05q4 Iraqi]Iraqi[/FREEBASE] deputy prime minister One of the things we are looking for is to give people a reason to come to [FREEBASE mid=/m/019rl6 Yahoo]Yahoo[/FREEBASE] to try search," said [FREEBASE mid=/m/075wwjc Tim Mayer]Tim Mayer[/FREEBASE], product vice president for [FREEBASE mid=/m/05h7w3 Yahoo Search]Yahoo Search[/FREEBASE]

### notes: throw out data if the text between two anchors is longer than some
### threshold, because it is likely to be noise at that point

def process(data_chunk, name):
	segment = 0
	f = open(data_path + name + '.processed_' + str(segment), 'w')
	counter, skipped = 0, 0
	output = []
	for i, line in enumerate(data_chunk):
		counter += 1
		reversd = 0
		# if counter > 100000:
		# 	break
		line = line.strip().split('\t')
		# print line
		relationships, leftEntityID, rightEntityID = line[0].split('..'), line[1], line[2]
		text = line[3]
		### parse out text between entity anchors
		start = '[FREEBASE mid=/m/' + leftEntityID.split('.')[1]
		end = '[FREEBASE mid=/m/' + rightEntityID.split('.')[1]
		
		try:
			start_regex = re.escape(start) + r"\s+(.*?)\](.*?)\[/FREEBASE\]"
			start_matched = re.search(start_regex, text).group(0)
			end_regex = re.escape(end) + r"\s+(.*?)\](.*?)\[/FREEBASE\]"
			end_matched = re.search(end_regex, text).group(0)
			
			leftEntity = re.search(r'\](.*?)\[/', start_matched).group(1)
			rightEntity = re.search(r'\](.*?)\[/', end_matched).group(1)
			# print start_matched, end_matched, leftEntity, rightEntity

			if (text.index(start_matched) < text.index(end_matched)):
				result = (text.split(start_matched))[1].split(end_matched)[0] 
			else: ### else reverse what left and right entities are
				result = (text.split(end_matched))[1].split(start_matched)[0] 
				temp = leftEntity
				leftEntity = rightEntity
				rightEntity = temp
				
				temp = leftEntityID
				leftEntityID = rightEntityID
				rightEntityID = temp
				reversd = 1
			result = re.sub("[\(\[].*?[\)\]]", "", result).strip() # string btw anchors
		except:
			skipped += 1
			continue
		result = [relationships, reversd, (leftEntity, leftEntityID), result, (rightEntity, rightEntityID)]
		# string = '###'.join(map(lambda x: str(x), result)) + '\n'
		output.append(str(result) + '\n')

		if i % 1000000 == 0 and i > 0: ### buffer strings to output, write to file 
			for o in output:
				f.write(o)
			f.close()
			output = []
			print '\t' + str(i) + ' examples done'
			print '\tread ' + str(counter) + ' lines with ' + str(skipped) + ' skipped lines'
			segment += 1
			f = open(data_path + name + '.processed_' + str(segment), 'w')

	### finish last few lines
	for o in output:
		f.write(o)
	f.close()
	output = []
	print '\t' + str(i + len(output)) + ' examples done'
	print name + ': read ' + str(counter) + ' lines with ' + str(skipped) + ' skipped lines'

data_path = '/Users/corbinrosset/Desktop/QA_datasets/VanDurme/annotated_clueweb/ClueWeb09_English_1/'

# corpora = ['en0000.gz', 'en0001.gz', 'en0002.gz', 'en0003.gz', \
# 	'en0004.gz', 'en0005.gz', 'en0006.gz', 'en0007.gz', 'en0008.gz', \
# 	'en0009.gz', 'en0010.gz']

corpora = ['en0001', 'en0002', 'en0003', 'en0004', 'en0005', 'en0006', \
	'en0007', 'en0008', 'en0009', 'en0010']

for corpus in corpora:
	try:
		data_chunk = open(data_path + corpus, 'r')
		print 'processing: ' + str(corpus)
		process(data_chunk, corpus)
	except:
		print 'ERROR: corpus does not exist or may not be unzipped...'
		continue



