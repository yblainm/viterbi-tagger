import nltk
from collections import defaultdict

# print(nltk.corpus.brown.tagged_words(tagset='universal')[:25])
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


def viterbi(inputseq, transprobs, obsprobs, statenums, states=tags):

	sequence = ["."] + inputseq
	trellis = defaultdict(lambda: defaultdict(lambda: list()))
	trellis[0]['.'] = [1, None]

	for i in range(1, len(sequence)):
		for state in states:
			for past_state, v in trellis[i-1].items():
				if not (past_state, state) in transprobs:
					transprobs[(past_state, state)] = 1.0/(statenums[past_state]+2)

				if not (state, sequence[i]) in obsprobs:
					obsprobs[(state, sequence[i])] = 1.0/(statenums[state]+2)

				prob = trellis[i-1][past_state][0] * transprobs[(past_state, state)] * obsprobs[(state, sequence[i])]

				if state in trellis[i]:
					if prob > trellis[i][state][0]:
						trellis[i][state] = [prob, past_state]
				else:
					trellis[i][state] = [prob, past_state]

	tail = max(trellis[len(sequence)-1].items(), key=lambda x: x[1][0])

	mle = [tail[0]]

	for i in range(0, len(sequence)-2):
		tail = (tail[1][1], trellis[len(sequence)-i-1][tail[1][1]])
		mle = trellis[len(sequence)-i-1][tail[0]][1:2] + mle
		#print(i, tail)

	return [x for x in zip(inputseq, mle)]


if __name__ == "__main__":
	training_corpus = nltk.corpus.brown.tagged_words(tagset='universal')
	tag_nums, trans_nums, obs_nums, trans_prob, obs_prob = train(training_corpus)
	print(viterbi(nltk.tokenize.word_tokenize("This is a test."), trans_prob, obs_prob, tag_nums))
