

class Graph():
	'''
		A class to store and provide access to multi-relational data

		The intention is that separate KGs will be made for KB and textual triples

		Based on a couple of similar papers:

		- Recurrent TransE of "Composing Relationships with Translations" by
		  Alberto Garcia-Duran, Antoine Bordes, and Nicolas Usunier
			They focused on “unambiguous” paths, so that the reasoning might 
			actually make sense. In particular, they considered only paths 
			where r1 is either a 1-to-1 or a 1-to-M relationship, and where 
			r2 is either a 1-to-1 or a M-to-1 relationship. In their 
			experiments, the paths created for training only consider the 
			training subset of facts

		- "Compositional Learning of Embeddings for Relation Paths in 
			Knowledge Bases and Text"
		- "Compositional Vector Space Models for Knowledge Base Completion"


	'''
	
	def __init__(self, numEntities, numRels, true_triples, text_quads = False):
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

		self.outgoing  = [] # outgoing[i] = set([set of all (entity, rel) pairs that i is 
						    # incident on via relation rel]) 
		self.incoming  = [] # same as above, except incoming to i
		self.relations = [] # relations[r] = set([set of all (e_h, e_t) pairs
							# for which e_h connects to e_t via relation r])
		# for triple in triples ...blah blah

	def addLink(self, rindex, eindex1, eindex2):
		'''
			add a link from entity 1 to 2 whose type is specified by rindex 
		'''
		pass

	def getIncoming(self, eindex):
		pass

	def getOutgoing(self, eindex):
		pass
	
	def existsLink(self, e_h, e_t, relation=None):
		'''
			optionally specify which relation
		'''
		pass
	
	def existsPath(self, e_h, e_t, length):
		'''
			does BFS from e_h for at most length iterations, returns whether e_t 
			reachable from e_h
		'''
	def generateRandomPath(self, from, length, to = False):
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
		pass

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



	

	

	

	
