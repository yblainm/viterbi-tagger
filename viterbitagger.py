import nltk
from collections import defaultdict

print(nltk.corpus.brown.tagged_words(tagset='universal')[:25])
tags = ('VERB','NOUN','PRON','ADJ','ADV','ADP','CONJ','DET','NUM','PRT','X','.')

def train(corpus):
	tag_nums = defaultdict(int)
	trans_nums = defaultdict(int)
	obs_nums = defaultdict(int)
	# trans_prob = defaultdict(float)
	# obs_prob = defaultdict(float)

	tag_nums['.'] = 1
	tag_nums[corpus[0][1]] = 1
	trans_nums[('.', corpus[0][1])] = 1
	obs_nums[('.','.')] = 1

	for i in range(1, len(corpus)):
		pair = corpus[i-1:i+1]
		tag_nums[pair[0][1]] +=1
		trans_nums[(pair[0][1], pair[1][1])] += 1
		obs_nums[(pair[0][1],pair[0][0])] += 1

	trans_prob = {k: v/tag_nums[k[0]] for k,v in trans_nums.items()}
	obs_prob = {k: v/tag_nums[k[0]] for k,v in obs_nums.items()}

	return tag_nums, trans_nums, obs_nums, trans_prob, obs_prob


if __name__ == "__main__":
	training_corpus = nltk.corpus.brown.tagged_words(tagset='universal')
	tag_nums, trans_nums, obs_nums, trans_prob, obs_prob = train(training_corpus)
	print(tag_nums, trans_nums)

