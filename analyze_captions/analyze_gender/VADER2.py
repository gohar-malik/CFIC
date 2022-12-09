from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json


# load 4 result files
f 					= open('results/val2014_dark_mod_coco.json')
dark_mod_coco 		= json.load(f)
f 					= open('results/val2014_dark_mod_conceptual-captions.json')
dark_mod_cc 		= json.load(f)
f 					= open('results/val2014_dark_orig_coco.json')
dark_orig_coco 		= json.load(f)
f 					= open('results/val2014_dark_orig_conceptual-captions.json')
dark_orig_cc 		= json.load(f)


f_list 				= [dark_mod_coco, dark_mod_cc,  dark_orig_coco, dark_orig_cc]
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