import json

# load profane words
f = open('profane_words.json')
profane_words = json.load(f)

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
profane_count_list 	= []



for file in f_list:
	profane_count = 0
	for key in file:
		sentence = file[key]
		words = sentence.split(' ')
		for word in words:
			if word in profane_words:
				profane_count += 1
				break
	profane_count_list.append(profane_count)

print(profane_count_list)