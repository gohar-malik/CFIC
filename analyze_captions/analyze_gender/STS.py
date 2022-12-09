import json
import time
from huggingface_hub.inference_api import InferenceApi
inference = InferenceApi(repo_id="sentence-transformers/stsb-roberta-base-v2", token="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")


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

text_similarity_list = []


for i,orig_file in enumerate(f_list_orig):
	mod_file = f_list_mod[i]

	total_similarity_score = 0

	for j,orig_key in enumerate(orig_file):
		mod_key = list(mod_file.keys())[j]
		orig_sentence = orig_file[orig_key]
		mod_sentence = mod_file[mod_key]
		inf_score = inference(inputs={"source_sentence": orig_sentence, "sentences": [mod_sentence]})
		total_similarity_score += inf_score[0]

	text_similarity_list.append(total_similarity_score/len(mod_file))


print(text_similarity_list)