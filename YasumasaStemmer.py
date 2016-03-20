'''
The problem of lemmatization is often attacked algorithmically.

I personally think this is a mistake in syntatic processing (not
natural language processing). There aren't that many words, and
there aren't really that many plurals, posessions, intransitives, etc.

In such a limited space, I think we can hold this in memory.

Consider common failures of the Porter Stemmer or the
Lancaster Stemmer:

have -> hav
decide -> decid
police -> pol

Why not just enumerate specific relationships for common words
as opposed to trying to divine the stem?

A huge thank you to Yasumasa Someya. In 1998, he made a great
map that is thorough and is just what is needed:

have <- has,having,had,'d,'ve,d,ve
decide <- decides,deciding,decided
police <- polices,policing,policed

1998 may seem some time ago now but thankfully, the delta-t on the
English language is small.

The file e_lemma.txt courtesy of:
	http://lexically.net/downloads/BNC_wordlists/e_lemma.txt
'''

class YasumasaStemmer():
	'''
	Keep in mind how fickle this can be:
	When covering the 2016 election, sander -> sanders. . .
	'''
	def __init__(self):
		self.stems = {}
		with open('e_lemma.txt','r') as stems:
			for line in stems:
				line = line.replace('\n','')
				parsed = line.split(' -> ')
				alts = parsed[1].split(',')
				for a in alts:
					self.stems[a] = parsed[0]

	def stem(self,word):
		if word in self.stems.keys():
			return self.stems[word]
		else:
			return word