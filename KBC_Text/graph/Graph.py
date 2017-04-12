from random import randint, sample
from KBC_Text.utils.Utils import *


class Graph():
	'''
		A class to store and provide access to multi-relational data

		The intention is that separate KGs will be made for KB and textual triples

		Based on a couple of similar papers:

		- Recurrent TransE of "Composing Relationships with Translations" by
		  Alberto Garcia-Duran, Antoine Bordes, and Nicolas Usunier
		- "Compositional Learning of Embeddings for Relation Paths in 
			Knowledge Bases and Text"
		- "Compositional Vector Space Models for Knowledge Base Completion"


	'''
	
	def __init__(self, state, true_triples, text_quads = False):
		'''
			Construct adjacency list data structures representing the KG
			
			All entities and rels must be indexed, that is, they must all be integers on
			[0, numEntities-1], and a corresponding index2entity mapping must exist to get
			back the original FB15k mids. 

			:param numEntities: number of possible entity indexes
			:param numRels: number of possible relation indexes
			:param true_triples: {(e_h, r_text, e_t)} iterable over all triples
			:param text_quads: {(e_h, r, r_text, e_t)} triples with indexes of which 
							   sentences mentioned this triple in Clueweb09 corpus
		'''
		state.logger.info('parameters:\n %s', str(state))
		start = time.clock()
		self.outgoing  = [set([]) for i in range(state.Nent)] 
		# outgoing[i] = set([set of all (rel, entity) pairs that i is 
		# incident on via relation rel]) 
		self.incoming  = [set([]) for i in range(state.Nent)] 
		# same as above, except {(entity, rel)} incoming to i
		self.relationsIn = [set([]) for i in range(state.Nrel)] 
		self.relationsOut = [set([]) for i in range(state.Nrel)]  
		# relationsIn[r] = set([set of all entity which are heads to
		# relation r])
		self.log = state.logger
		self.state = state
		self.batchSize = 100000
		self.triesPerPath = 20
		# for triple in triples ...blah blah
		if true_triples is not None:
			self.log.info("%s triples in KB, initializing graph..." % \
				len(true_triples))
			for (h, r, t) in true_triples:
				self.checkEntityIdx(h)
				self.checkEntityIdx(t) 
				self.checkRelationIdx(r)
				self.addLink(h, r, t)

		elapsed = time.clock()
		elapsed = elapsed - start
		self.log.info("constructed graph data structures in: %s seconds" % elapsed)

	def checkEntityIdx(self, e):
		assert e < self.state.Nent
		assert e >= 0

	def checkRelationIdx(self, r):
		assert r < self.state.Nrel
		assert r >= 0

	def addLink(self, eindex1, rindex, eindex2):
		'''
			add a link from entity 1 to 2 whose type is specified by rindex 
		'''
		self.outgoing[eindex1].add((rindex, eindex2))
		self.incoming[eindex2].add((eindex1, rindex))
		self.relationsIn[rindex].add(eindex1)
		self.relationsOut[rindex].add(eindex2)


	def getIncoming(self, tailindex, rindex=None):
		'''
			if rindex is not set, return all {(headindex, rindex)}
			which make (headindex, rindex, tailindex) true (theyre in the KB)

			if rindex is set, return the set of all {headentities}
			that make (headindex, rindex, tailindex) true
		'''
		self.checkEntityIdx(tailindex)
		if not rindex:
			return self.incoming[tailindex] 
		else:
			### this is probably most efficient since most relations are
			### not many-to-many, so widdling down which entities have
			### rindex as an outgoing edge is a good idea
			self.checkRelationIdx(rindex)
			out = []
			for headidx in self.relationsIn[rindex]:
				if (rindex, tailindex) in self.outgoing[headidx]:
					out.append(headidx)

			return out

	def getOutgoing(self, headindex, rindex=None):
		'''
			similar to getIncoming:
		'''
		self.checkEntityIdx(headindex)
		if not rindex:
			return self.outgoing[headindex] 
		else:
			### this is probably most efficient since most relations are
			### not many-to-many, so widdling down which entities have
			### rindex as an outgoing edge is a good idea
			self.checkRelationIdx(rindex)
			out = []
			for tailidx in self.relationsOut[rindex]:
				if (headindex, rindex) in self.incoming[headindex]:
					out.append(tailidx)
			return out
	
	def existsLink(self, e_h, e_t, rindex=None):
		'''
			optionally specify which relation
		'''
		self.checkEntityIdx(e_h)
		self.checkEntityIdx(e_t)
		if not rindex:
			out = []
			for r in range(self.state.Nrel):
				if e_h in self.relationsIn[r] and e_t in self.relationsOut[r]:
					out.append(r)
			return out if out else False
		else:
			self.checkRelationIdx(rindex)
			if (rindex, e_t) in self.outgoing[e_h]:
				assert (e_h, rindex) in self.incoming[e_t]
				return True
	
	def existsPath(self, e_h, e_t, length, rindxs=None):
		'''
			does BFS from e_h for at most length iterations, returns whether e_t 
			reachable from e_h
		'''
		if length <= 0:
			return False
		elif rindxs:
			assert hasattr(rindxs, '__iter__')
			assert len(rindxs) == length

			pair = (e_h, rindxs[0])
			for i in range(1, len(rindxs)):
				tail = self.outgoing.get((pair), None)
				if tail == None:
					return False
				else:
					pair = (tail, rindxs[i])
			return True if tail == e_t else False
		elif length == 1: # (and implicitly not rindxs)
			return any((e_h in self.relationsIn[r] and e_t in self.relationsOut[r]) for r in range(self.state.Nrel))
		else:
			raise NotImplementedError
			### do a BFS...

	def random_entity(self):
		return randint(0, self.state.Nent-1)

	def random_relation(self):
		return randint(0, self.state.Nrel-1)

	def random_entry(self, lst):
		idx = randint(0, len(lst)-1)
		return lst[idx]

	def sample_excluding(self, lst, excl):
		assert isinstance(lst, set)
		try:
			x = sample(lst - set(excl), 1)
		except ValueError:
			return None # usually means sampling from empty set!
		return x


	def generateRandomPath(self, start, length, to = False):
		'''
			if to == True, generate a random path to e of the specified length,
			else generate a path starting at e.

			Be very careful here. The most helpful paths follow "horn-rule" 
			constraints, that is, if e1 -- r --> e2 is a triple (horn-rule 
			head), it is helpful to generate paths of length L from 
			e1 -- / -- / --> e2 which represent the body of the horn rule

			To do this, specify the "to" argument, and make sure that 
			"from" and "to" are adjacent! That is your responsibility!
		'''
		# TODO: store/calculate transition probabilities to facilitate this
		assert length >= 1
		if not to:
			headidx, out = start, [(None, start)]
			l, tries = 0, 0
			while l < length and tries < self.triesPerPath:
				x = self.sample_excluding(self.outgoing[headidx], out)
				if x == None: # rollback one edge bc we are at dead end
					out = out[:-1]
					headidx = out[-1][1]
					l = max(0, l-1)
					tries += 1
					# self.log.warning('\trolling back path one edge')
					continue
				(relidx, tailidx) = x[0] # sample() wraps output in list
				out.append((relidx, tailidx))
				headidx = tailidx
				l += 1
			assert (len(out) == length + 1) or (tries >= self.triesPerPath)
			return out if tries < 100 else None
		else:
			raise NotImplementedError

	def pathToString(self, path):
		if self.state.useHornPaths and self.state.needIntermediateNodesOnPaths:
			# format of path: (inteded rel, [(None, h), (r1, e1), ... (rn, t)])
			return str(path[0]) + '\t' + str(path[1])
		elif not self.state.useHornPaths and self.state.needIntermediateNodesOnPaths: 
			# format of path: [(None, h), (r1, e1), ... (rn, t)]
			return str(path)

		elif self.state.useHornPaths and not self.state.needIntermediateNodesOnPaths:
			# format of path: (inteded rel, [(None, h), (r1, e1), ... (rn, t)])
			# transform to (intended rel, h, t, [r1, r2, ..., rn])
			return str(path[0]) + '\t' + str(path[1][0][1]) + '\t' + str(path[1][-1][1]) + str(map(lambda (x, y): x, path[1][1:]))

		else:
			# format of path: [(None, h), (r1, e1), ... (rn, t)]
			# transform to (h, t, [r1, r2, ..., rn])
			return str(path[0][1]) + '\t' + str(path[-1][1]) + '\t' + str(map(lambda (x, y): x, path[1:]))

	def generatePathDataset(self):
		self.log.info('BEGIN RANDOM PATH GENERATION')
		self.log.info('path files saved in directory: %s', self.state.savepath + self.state.identifier + '/')
		assert len(self.state.path_lengths) == len(self.state.num_paths)

		for length, numPaths in zip(self.state.path_lengths, self.state.num_paths):
			fileId = 'length_' + str(length) + '_numPaths_' + str(numPaths) + '.path'
			self.log.info('generating %s paths of length %s and saving them to %s' % (numPaths, length, fileId))
			paths = []
			count = 0
			while count < numPaths:
				start = self.random_entity()
				while len(self.outgoing[start]) == 0: #start with connected nod
					start = self.random_entity()
				if self.state.useHornPaths:
					# pick random adjecent 
					(rel, tail) = sample(self.outgoing[start])
					path = self.generateRandomPath(start, length, to=tail)
					if path:
						paths.append((rel, path)) # path represents the rel
						count += 1
					else:
						# self.log.warning('\tfailed to produce path this try')
						continue
				else: # there is no "closing" edge bc it's a path not cycle
					path = self.generateRandomPath(start, length)
					if path:
						paths.append(path)
						count += 1
					else:
						# self.log.warning('\tfailed to produce path this try')
						continue
				# print self.pathToString(path)
				if count % self.batchSize == 0:
					self.log.info('\twriting batch of %s paths to file' % self.batchSize)
					f = open(self.state.savepath + self.state.identifier + '/' + fileId, 'a+')
					for path in paths:
						f.write(self.pathToString(path) + '\n')
					paths = []
			if paths: # if any remaining paths...
				f = open(self.state.savepath + self.state.identifier + '/' + fileId, 'a+')
				for path in paths:
					f.write(self.pathToString(path) + '\n')
		self.log.info('Done generating path datasets')


					




	 # 		def random_path_query(self, length):
	 #	 	'''
     #			from Kelvin Gu
	 #		'''
	 #        while True:
	 #            # choose initial entity uniformly at random
	 #            source = self.random_entity()

	 #            # sample a random walk
	 #            path, target = self.random_walk(source, length)

	 #            # Failed to find random walk. Try again.
	 #            if path is None:
	 #                continue

	 #            pq = PathQuery(source, path, target)
	 #            return pq


  #		def random_walk(self, start, length, no_return=False):
  # 	'''
  #		 	from Kelvin Gu
  #		'''
  #       max_attempts = 1000
  #       for i in range(max_attempts):

  #           sampled_path = []
  #           visited = set()
  #           current = start
  #           for k in range(length):
  #               visited.add(current)

  #               r = random.choice(self.neighbors[current].keys())
  #               sampled_path.append(r)

  #               candidates = self.neighbors[current][r]

  #               if no_return:
  #                   current = util.sample_excluding(candidates, visited)
  #               else:
  #                   current = random.choice(candidates)

  #               # no viable next step
  #               if current is None:
  #                   break

  #           # failed to find a viable walk. Try again.
  #           if current is None:
  #               continue

  #           return tuple(sampled_path), current

  #       return None, None

	def size(self):
		'''
			how much memory is each of outgoing, incoming, relations consuming?
		'''
		pass

	def partitionRels(self):
		'''
			after initialization, partition the relationships into 1-to-1, 
			1-to-M, M-to-1, and M-to-M based on the ~1.2 threshold for average
			number of head (resp. tail) entity types. 

			TODO: do we have entity type information???

			from TATEC paper: https://arxiv.org/pdf/1506.00999.pdf
			A relationship is considered as 1-to-1, 1-to-M, M-to-1 or M-M regarding the variety of arguments head given a tail and vice versa. If the average number of different heads for the whole set of unique pairs (label, tail) given a relationship is below 1.5 we have considered it as 1, and the same in the other way around. The number of relations classified as 1-to-1, 1-to-M, M-to-1 and M-M is 353, 305, 380 and 307, respectively


		'''
		pass

	# def relation_stats(self):
	# 	'''
	# 		from Kelvin Gu
	# 	'''
 #        stats = defaultdict(dict)
 #        rel_counts = Counter(r for s, r, t in self.triples)

 #        for r, args in self.relation_args.iteritems():
 #            out_degrees, in_degrees = [], []
 #            for s in args['s']:
 #                out_degrees.append(len(self.neighbors[s][r]))
 #            for t in args['t']:
 #                in_degrees.append(len(self.neighbors[t][invert(r)]))

 #            domain = float(len(args['s']))
 #            range = float(len(args['t']))
 #            out_degree = np.mean(out_degrees)
 #            in_degree = np.mean(in_degrees)
 #            stat = {'avg_out_degree': out_degree,
 #                    'avg_in_degree': in_degree,
 #                    'min_degree': min(in_degree, out_degree),
 #                    'in/out': in_degree / out_degree,
 #                    'domain': domain,
 #                    'range': range,
 #                    'r/d': range / domain,
 #                    'total': rel_counts[r],
 #                    'log(total)': np.log(rel_counts[r])
 #                    }

 #            # include inverted relation
 #            inv_stat = {'avg_out_degree': in_degree,
 #                        'avg_in_degree': out_degree,
 #                        'min_degree': stat['min_degree'],
 #                        'in/out': out_degree / in_degree,
 #                        'domain': range,
 #                        'range': domain,
 #                        'r/d': domain / range,
 #                        'total': stat['total'],
 #                        'log(total)': stat['log(total)']
 #                        }

 #            stats[r] = stat
 #            stats[invert(r)] = inv_stat

 #        return stats


if __name__ == '__main__':
	state = DD()
	state.path_lengths = [2, 3, 4, 5]
	state.num_paths = [50000000, 50000000, 50000000, 50000000]
	state.useHornPaths = False
	state.needIntermediateNodesOnPaths = False
	state.savepath = '/Users/corbinrosset/Dropbox/Arora/QA-code/src/KBC_Text/data/GraphPaths/'
	state.identifier = 'Paths'
	state.datapath = '/Users/corbinrosset/Dropbox/Arora/QA-code/src/KBC_Text/data/'
	state.dataset = 'FB15k'
	state.ntrain = 'all'
	state.nvalid = 'all'
	state.ntest = 'all'
	state.Nrel = 1345
	state.Nent = 14951
	
	if not os.path.isdir(state.savepath + state.identifier + '/'):
		os.makedirs(state.savepath + state.identifier + '/')
	state.logger, _ = initialize_logging(state.savepath + state.identifier + '/', state.identifier)

	_, _, _, trainlidx, trainridx, trainoidx, validlidx, validridx, validoidx, testlidx, testridx, testoidx, true_triples, entireKB = load_FB15k_data(state)

	trainKBmat = np.concatenate([trainlidx, trainoidx, trainridx]).reshape(3, \
				trainlidx.shape[0]).T
	trainKB = set([tuple(i) for i in trainKBmat]) # {(head, rel, tail)}

	# validKBmat = np.concatenate([validlidx, validoidx, validridx]).reshape(3, \
	# 			trainl.shape[0]).T
	# validKB = KB = set([tuple(i) for i in validKBmat]) # {(head, rel, tail)} 
	# testKBmat = np.concatenate([testlidx, testoidx, testridx]).reshape(3, \
	# 			trainl.shape[0]).T
	# testKB = KB = set([tuple(i) for i in testKBmat]) # {(head, rel, tail)} 


	g = Graph(state, trainKB, text_quads=False)
	g.generatePathDataset()
	
	

	

	

	
