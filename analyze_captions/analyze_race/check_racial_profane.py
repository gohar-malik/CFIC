import json

# load profane words
f 					= open('profane_words.json')
profane_words 		= json.load(f)
racial_descriptors	= ["white", "Caucasian", "Black", "African", "Asian", "Latino", "Latina", "Latinx", "Hispanic", "Native", "Indigenous"]

for i in range(len(racial_descriptors)):
	racial_descriptors[i] = racial_descriptors[i].lower()


# load 4 result files
f 					= open('../captions_stgan_race/val2014_dark_mod_coco.json')
dark_mod_coco 		= json.load(f)
f 					= open('../captions_stgan_race/val2014_dark_mod_conceptual-captions.json')
dark_mod_cc 		= json.load(f)
f 					= open('../captions_stgan_race/val2014_dark_orig_coco.json')
dark_orig_coco 		= json.load(f)
f 					= open('../captions_stgan_race/val2014_dark_orig_conceptual-captions.json')
dark_orig_cc 		= json.load(f)
# [26, 11, 26, 19]
# w:18 b:8
# w:4 b:7
# w:20 b:6
# w:10 b:9


f_list 				= [dark_mod_coco, dark_mod_cc, dark_orig_coco, dark_orig_cc]
profane_count_list 	= []
profane_key_list 	= []
racial_count_list	= []
racial_key_list		= []



for file in f_list:
	profane_count, racial_count, profane_key, racial_key = 0, 0, [], []
	# print(len(file)) (332)
	for key in file:
		sentence = file[key]
		words = sentence.split(' ')
		for word in words:
			if word.lower() in profane_words:
				profane_count += 1
				profane_key.append(key)
			if word.lower() in racial_descriptors:
				racial_count += 1
				racial_key.append(key)
				print(word)
	profane_count_list.append(profane_count)
	profane_key_list.append(profane_key)
	racial_count_list.append(racial_count)
	racial_key_list.append(racial_key)
	print('\n\n')


# print results
print('profane_count_list',profane_count_list)
print("***************")
print('dark->light_coco')
for key in profane_key_list[0]:
	print(key)
print('dark->light_cc')
for key in profane_key_list[1]:
	print(key)
print('dark_coco')
for key in profane_key_list[2]:
	print(key)
print('dark_cc')
for key in profane_key_list[3]:
	print(key)
print('****************')


print('racial count list',racial_count_list)
print("***************")
print('dark->light_coco')
for key in racial_key_list[0]:
	print(key)
print('dark->light_cc')
for key in racial_key_list[1]:
	print(key)
print('dark_coco')
for key in racial_key_list[2]:
	print(key)
print('dark_cc')
for key in racial_key_list[3]:
	print(key)
print('****************')