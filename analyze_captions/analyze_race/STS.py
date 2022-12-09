import json
import time
from huggingface_hub.inference_api import InferenceApi
inference = InferenceApi(repo_id="sentence-transformers/stsb-roberta-base-v2", token="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")


# load 4 result files
f 					= open('../captions_stgan_race/val2014_dark_mod_coco.json')
dark_mod_coco 		= json.load(f)
f 					= open('../captions_stgan_race/val2014_dark_mod_conceptual-captions.json')
dark_mod_cc 		= json.load(f)
f 					= open('../captions_stgan_race/val2014_dark_orig_coco.json')
dark_orig_coco 		= json.load(f)
f 					= open('../captions_stgan_race/val2014_dark_orig_conceptual-captions.json')
dark_orig_cc 		= json.load(f)


f_list_orig 		= [dark_orig_coco, dark_orig_cc]
f_list_mod			= [dark_mod_coco, dark_mod_cc]


text_similarity_dict 	= dict()
text_similarity_avg 	= list()


for i,orig_file in enumerate(f_list_orig):
	mod_file = f_list_mod[i]

	similarity_score_list = []
	total_score = 0

	for j,orig_key in enumerate(orig_file):
		print(f'{j} / {len(orig_file)}')
		orig_sentence 	= orig_file[orig_key]
		mod_sentence 	= mod_file[orig_key]
		
		# get STS scores
		inf_score = inference(inputs={"source_sentence": orig_sentence, "sentences": [mod_sentence]})
		
		similarity_score_list.append(inf_score[0])
		total_score += inf_score[0]

	if i == 0:
		text_similarity_dict['race_coco'] = similarity_score_list
	else:
		text_similarity_dict['race_cc'] = similarity_score_list
	text_similarity_avg.append(total_score/len(orig_file))


with open('STS_result.json','w+') as f:
	json.dump(text_similarity_dict, f)
print('write all similarity results to STS_result.json')
print('race counterfactual, coco trained, captions semantic similarity               ', text_similarity_avg[0])
print('race counterfactual, conceptual-captions trained, captions semantic similarity', text_similarity_avg[1])