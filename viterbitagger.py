import nltk
from itertools import islice

#print(nltk.corpus.brown.tagged_words()[:25])

def train(corpus):
	trans = {}
	obs = {}
	for pair in islice(corpus, len(corpus)):
		pass
	return None, None


if __name__ == "__main__":
	training_corpus = nltk.corpus.brown.tagged_words(tagset='universal')
	trans, obs = train(training_corpus)

