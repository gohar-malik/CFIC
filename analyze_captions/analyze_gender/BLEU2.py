from nltk.translate.bleu_score import sentence_bleu


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

f_list_orig 		= [dark_orig_coco, dark_orig_cc]
f_list_mod			= [dark_mod_coco, dark_mod_cc]

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