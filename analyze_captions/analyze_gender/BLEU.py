from nltk.translate.bleu_score import sentence_bleu


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


f_list_orig 		= [female_orig_coco, female_orig_cc, male_orig_coco, male_orig_cc]
f_list_mod			= [female_mod_coco, female_mod_cc, male_mod_coco, male_mod_cc]

bleu_score_list = []

for i,orig_file in enumerate(f_list_orig):
	mod_file = f_list_mod[i]

	total_score = 0

	for j,orig_key in enumerate(orig_file):
		mod_key = list(mod_file.keys())[j]
		orig_sentence = orig_file[orig_key]
		mod_sentence = mod_file[mod_key]
		total_score += sentence_bleu([orig_sentence.split()], mod_sentence.split())

	bleu_score_list.append(total_score/len(mod_file))


print(bleu_score_list)