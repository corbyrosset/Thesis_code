import os
'''a file to standardize the format of all text similarity files so that every 
	line contains 3 tab-separated elements: the first two are phrases that may 
	or may not be similar, and the last element is a score [0, 5] rating how 
	similar they are. 

	The data was obtained from https://github.com/brmson/dataset-
	sts/tree/master/data/sts/semeval-sts for years 2012-2016 and comprises of 
	the test/dev evaluation sets. (Forked data to https://github.com/corbyrosset/dataset-sts)

	make sure to remove *_normed files before running this script, or you'll
	get *_normed_normed files

	Also outputs a combined_test_data_normed file which appends all outputs. 
	It has 9858 lines about. 
'''

def isfloat(x):
    try:
        a = float(x)
    except ValueError:
        return False
    else:
        return True

combined = []
for filename in os.listdir('.'):
	if filename.startswith('.') or filename.endswith('.py'):
		continue # ignore hidden files
	newfile = open(filename + '_normed', 'w')
	print 'processing ' + str(filename)
	with open(filename, 'r') as f:
		for line in f:
			line = filter(lambda x: x != '' and x != '\n', line.rstrip("\n").split('\t'))
			print line
			if not line:
				continue
			if len(line) == 2 and (not isfloat(line[0])) and not (isfloat(line[1])):
				continue ### skip lines that don't have a score
			elif len(line) == 3 and (not isfloat(line[0])) and not (isfloat(line[1])) and (isfloat(line[2])):
				newfile.write(('\t').join(line) + '\n')
			elif len(line) == 3 and (isfloat(line[0])) and not (isfloat(line[1])) and (not isfloat(line[2])):
				temp = line[0]
				line[0] = line[2]
				line[2] = temp
				newfile.write(('\t').join(line) + '\n')
				combined.append(('\t').join(line))
			else:
				print 'error, ' + str(filename) + ' not formatted properly: ' 
				print  line
				exit(1)
	newfile.close()

c = open('combined_test_data_normed', 'w')
for line in combined:
	c.write(line + '\n')
c.close()

