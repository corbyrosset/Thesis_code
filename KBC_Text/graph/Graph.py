

class Graph():
	'''
		A class to store and provide access to multi-relational data

		The intention is that separate KGs will be made for KB and textual triples
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
		self.relations = [] # relations[r] = set([set of all (e_h, e_t) pairs for which e_h
							# connects to e_t via relation r])

	
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
	def generateRandomPath(self, e, length, to = False)
		'''
			if to == True, generate a random path to e of the specified length,
			else generate a path starting at e.
		'''
		# TODO: store/calculate transition probabilities to facilitate this
		pass
	

	

	

	
