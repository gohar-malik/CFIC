import json

results = json.load(open('STS_result.json','r'))

coco_result_list 	= results['race_coco']
cc_result_list 		= results['race_cc']

f 					= open('../captions_stgan_race/val2014_dark_mod_coco.json')
coco_f_coun 		= json.load(f)
f 					= open('../captions_stgan_race/val2014_dark_orig_coco.json')
coco_f_orig 		= json.load(f)
f 					= open('../captions_stgan_race/val2014_dark_mod_conceptual-captions.json')
cc_f_coun 			= json.load(f)
f 					= open('../captions_stgan_race/val2014_dark_orig_conceptual-captions.json')
cc_f_orig 			= json.load(f)


coco_dict 	= dict()
cc_dict 	= dict()

# add key for every similarity scores
for i,key in enumerate(coco_f_orig):
	coco_dict[key] = coco_result_list[i]

for i,key in enumerate(cc_f_orig):
	cc_dict[key] = cc_result_list[i]

# sort similarity score to find smallest ones
coco_dict 	= dict(sorted(coco_dict.items(), key=lambda item: item[1]))
cc_dict 	= dict(sorted(cc_dict.items(), key=lambda item: item[1]))

# print
print('**********captions with least STS scores****************')
print('\n ######### coco ###############)')
for i, key in enumerate(coco_dict):
	if i == 10:
		break
	print(f'{key}:  orig captions: {coco_f_orig[key]},\n                            coun captions: {coco_f_coun[key]},  score: {coco_dict[key]}')


print('\n ######### cc ###############)')
for i, key in enumerate(cc_dict):
	if i == 10:
		break
	print(f'{key}:  orig captions: {cc_f_orig[key]},\n                            coun captions: {cc_f_coun[key]},  score: {cc_dict[key]}')