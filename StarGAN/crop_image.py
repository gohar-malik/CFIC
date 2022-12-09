import image_slicer
import os
import sys

output_image_index = 1
output_image_path = f"stargan_celeba_128/results/{output_image_index}-images.jpg"


path = 'data/mydata/test/Female'
input_female_files 	= os.listdir(path)

path = 'data/mydata/test/Male'
input_male_files 	= os.listdir(path)

input_female_index 	= 0
input_male_index 	= 0

save_dir 	= 'output/Female'	# would change for male
save_index 	= 4					# would change for male 
male_flag 	= False



while os.path.exists(output_image_path): 
	print(output_image_path)

	# for every output image, crop image into single pieces
	tiles = image_slicer.slice(output_image_path, row=16, col=6, save=False)
	
	# only save one target image for one column
	new_tiles = tuple()
	for index,tile in enumerate(tiles):
		if index%6==save_index:
			new_tiles = (*new_tiles, tile)
	image_slicer.save_tiles(new_tiles, directory=save_dir, format='png')
	
	# rename saved file
	for i in range(16):

		# detect male change, this condition only enter once 
		if male_flag is False and input_female_index == len(input_female_files):
			male_flag = True
			save_index = 5
			save_dir = 'output/Male'
			# save file in new dir
			new_tiles = tuple()
			for index,tile in enumerate(tiles):
				if index%6==save_index:
					new_tiles = (*new_tiles, tile)
			image_slicer.save_tiles(new_tiles, directory=save_dir, format='png')
		if input_male_index == len(input_male_files):
			sys.exit(0)

		# get current file name
		if i+1 < 10:
			tmp_str = '0'+str(i+1)
		else:
			tmp_str = str(i+1)
		cur_filename = os.path.join(save_dir,'_'+tmp_str+f'_0{save_index+1}.png')

		# get target filename
		if male_flag is False:
			tar_filename = input_female_files[input_female_index]
			input_female_index += 1
		else:
			tar_filename = input_male_files[input_male_index]
			input_male_index += 1
		tar_filename = os.path.join(save_dir, tar_filename.replace('jpg','png'))

		# rename file
		os.rename(cur_filename, tar_filename)

	# update output image path
	output_image_index += 1
	output_image_path = f"stargan_celeba_128/results/{output_image_index}-images.jpg"