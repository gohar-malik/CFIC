from nltk.translate.bleu_score import sentence_bleu
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


avg_bleu_score 	= 0
top_score_dict 	= {}
record_num		= 10

for i,orig_key in enumerate(f_orig):
	# counterfactual and original keys have same sequence
	mod_key 		= orig_key
	orig_sentence 	= f_orig[orig_key]
	mod_sentence 	= f_coun[mod_key]
	score           = sentence_bleu([orig_sentence.split()], mod_sentence.split())
	avg_bleu_score 	+= score

	# find pairs with smallest BLEU score
	if i < record_num:
		top_score_dict[orig_key] = score
	else:
		least_key = list(top_score_dict.keys())[record_num-1]
		if score < top_score_dict[least_key]:
			del top_score_dict[least_key]
			top_score_dict[orig_key] = score
			top_score_dict = dict(sorted(top_score_dict.items(), key=lambda item: abs(item[1])))

avg_bleu_score = avg_bleu_score/len(f_orig)


print(avg_bleu_score)
print('**********captions with least bleu scores****************')
for key in top_score_dict.keys():
	print(f'{key}: orig captions: {f_orig[key]}, \n                           coun captions: {f_coun[key]}, BLEU score: {top_score_dict[key]}')