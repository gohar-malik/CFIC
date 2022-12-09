from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json


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


f_list 				= [female_mod_coco, female_mod_cc,  female_orig_coco, female_orig_cc,
						male_mod_coco, male_mod_cc, male_orig_coco, male_orig_cc]
avg_sentiment_list  = []


analyzer = SentimentIntensityAnalyzer()


for file in f_list:
	total_sentiment = 0
	for key in file:
		sentence = file[key]
		vs = analyzer.polarity_scores(sentence)
		total_sentiment += vs['pos']
	avg_sentiment_list.append(total_sentiment/len(file))

print(avg_sentiment_list)