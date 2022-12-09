# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# sentences = ["A black man crying",
# 		"A white man smiling."]

# avg_sentiment_list  = []


# analyzer = SentimentIntensityAnalyzer()



# for sentence in sentences:
# 	vs = analyzer.polarity_scores(sentence)
# 	print(vs['pos'])
# 	print(vs['neg'])
# 	print(vs)


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json

dataset = 'cc'
print(f"*********for model trained by {dataset}*************")

if dataset == 'coco':
	# load caption files
	f 				= open('../captions_stgan_race/val2014_dark_mod_coco.json')
	f_coun 			= json.load(f)
	f 				= open('../captions_stgan_race/val2014_dark_orig_coco.json')
	f_orig 			= json.load(f)
elif dataset == 'cc':	
	f 				= open('../captions_stgan_race/val2014_dark_mod_conceptual-captions.json')
	f_coun 			= json.load(f)
	f 				= open('../captions_stgan_race/val2014_dark_orig_conceptual-captions.json')
	f_orig 			= json.load(f)
else:
	print('wrong dataset')
	exit()


neutral_sample_count_orig = 0
neutral_sample_count_coun = 0
neutral_intensity = 0.80

analyzer = SentimentIntensityAnalyzer()


for i,key in enumerate(f_orig):
	orig_caption = f_orig[key]
	coun_caption = f_coun[key]
	# sentimental analysis
	orig_vs = analyzer.polarity_scores(orig_caption)
	coun_vs = analyzer.polarity_scores(coun_caption)

	if orig_vs['neu'] > neutral_intensity:
		neutral_sample_count_orig += 1
	if coun_vs['neu'] > neutral_intensity:
		neutral_sample_count_coun += 1

print(f'the number of samples with neutral score > {neutral_intensity} in orginal       : {neutral_sample_count_orig/len(f_orig)}')
print(f'the number of samples with neutral score > {neutral_intensity} in counterfactual: {neutral_sample_count_coun/len(f_orig)}')
print(len(f_orig))