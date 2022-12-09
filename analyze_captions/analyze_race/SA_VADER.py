from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json

dataset = 'coco'
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


# first one is original and second one is counterfactual
avg_pos_score	= [0,0]
top_pos_dict	= {}
avg_neg_score	= [0,0]
top_neg_dict	= {}
record_num		= 10

analyzer = SentimentIntensityAnalyzer()


for i,key in enumerate(f_orig):
	orig_caption = f_orig[key]
	coun_caption = f_coun[key]
	# sentimental analysis
	orig_vs = analyzer.polarity_scores(orig_caption)
	coun_vs = analyzer.polarity_scores(coun_caption)
	avg_pos_score[0] += orig_vs['pos']
	avg_pos_score[1] += coun_vs['pos']
	avg_neg_score[0] += orig_vs['neg']
	avg_neg_score[1] += coun_vs['neg']

	# find pairs with largest SA score change
	pos_diff = abs(orig_vs['pos'] - coun_vs['pos'])
	neg_diff = abs(orig_vs['neg'] - coun_vs['neg'])
	if i < record_num:
		top_pos_dict[key] = pos_diff
		top_neg_dict[key] = neg_diff
	else:
		pos_least_key = list(top_pos_dict.keys())[record_num-1]
		if pos_diff > top_pos_dict[pos_least_key]:
			del top_pos_dict[pos_least_key]
			top_pos_dict[key] = pos_diff
			top_pos_dict = dict(sorted(top_pos_dict.items(), key=lambda item: -item[1]))

		neg_least_key = list(top_neg_dict.keys())[record_num-1]
		if neg_diff > top_neg_dict[neg_least_key]:
			del top_neg_dict[neg_least_key]
			top_neg_dict[key] = neg_diff
			top_neg_dict = dict(sorted(top_neg_dict.items(), key=lambda item: -item[1]))


avg_pos_score[0] = avg_pos_score[0]/len(f_orig)
avg_pos_score[1] = avg_pos_score[1]/len(f_coun)
avg_neg_score[0] = avg_neg_score[0]/len(f_orig)
avg_neg_score[1] = avg_neg_score[1]/len(f_coun)

print('avg positive socres of Original captions      ',avg_pos_score[0])
print('avg positive socres of Counterfactual captions',avg_pos_score[1])
print('avg negative socres of Original captions      ',avg_neg_score[0])
print('avg negative socres of Counterfactual captions',avg_neg_score[1])


print('**********captions different most in SA postive scores****************')
for key in top_pos_dict.keys():
	print(f'{key}: orig captions: {f_orig[key]}, \n                           coun captions: {f_coun[key]}, diff score: {top_pos_dict[key]}')


print('**********captions different most in SA negative scores****************')
for key in top_neg_dict.keys():
	print(f'{key}: orig captions: {f_orig[key]}, \n                           coun captions: {f_coun[key]}, diff score: {top_neg_dict[key]}')
