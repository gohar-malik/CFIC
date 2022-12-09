from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json

analyzer = SentimentIntensityAnalyzer()


# load 8 result files
f 					= open('results/val2014_female_mod_coco.json')
female_mod_coco 	= json.load(f)
f 					= open('results/val2014_female_mod_conceptual-captions.json')
female_mod_cc 		= json.load(f)
f 					= open('results/val2014_female_orig_coco.json')
female_orig_coco 	= json.load(f)
f 					= open('results/val2014_female_orig_conceptual-captions.json')
female_orig_cc 		= json.load(f)

f 					= open('results/val2014_male_mod_coco.json')
male_mod_coco 		= json.load(f)
f 					= open('results/val2014_male_mod_conceptual-captions.json')
male_mod_cc 		= json.load(f)
f 					= open('results/val2014_male_orig_coco.json')
male_orig_coco 		= json.load(f)
f 					= open('results/val2014_male_orig_conceptual-captions.json')
male_orig_cc 		= json.load(f)


f_list_orig 		= [female_orig_coco, female_orig_cc, male_orig_coco, male_orig_cc]
f_list_mod			= [female_mod_coco, female_mod_cc, male_mod_coco, male_mod_cc]

most_sentiment_list  = []



for i,orig_file in enumerate(f_list_orig):
	mod_file = f_list_mod[i]

	# dict place 10 samples with biggest sentimental change
	most_sentiment_sample_dict = dict()

	for j,orig_key in enumerate(orig_file):
		# get diff
		mod_key = list(mod_file.keys())[j]
		orig_sentence = orig_file[orig_key]
		mod_sentence = mod_file[mod_key]
		if i in [0,1] and 'woman' in mod_sentence.split(' '):
			continue
		if i in [2,3] and 'man' in mod_sentence.split(' '):
			continue
		orig_vs = analyzer.polarity_scores(orig_sentence)
		mod_vs = analyzer.polarity_scores(mod_sentence)
		diff = orig_vs['pos'] - mod_vs['pos']

		# update most sentiment dict by new diff
		if len(most_sentiment_sample_dict) < 20:
			most_sentiment_sample_dict[orig_key] = diff
			most_sentiment_sample_dict = dict(sorted(most_sentiment_sample_dict.items(), key=lambda item: -abs(item[1])))
		else:
			least_key = list(most_sentiment_sample_dict.keys())[19]
			if diff > most_sentiment_sample_dict[least_key]:
				del most_sentiment_sample_dict[least_key]
				most_sentiment_sample_dict[orig_key] = diff
				most_sentiment_sample_dict = dict(sorted(most_sentiment_sample_dict.items(), key=lambda item: -abs(item[1])))

	most_sentiment_list.append(most_sentiment_sample_dict)


print("*********************")
print('female_coco')
print(most_sentiment_list[0])
print('*********************\n')

print("*********************")
print('female_cc')
print(most_sentiment_list[1])
print('*********************\n')

print("*********************")
print('male_coco')
print(most_sentiment_list[2])
print('*********************\n')

print("*********************")
print('male_cc')
print(most_sentiment_list[3])
print('*********************\n')