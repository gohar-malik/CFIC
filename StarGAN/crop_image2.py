import image_slicer
import os
import sys

output_image_index = 1
output_image_path = f"stargan_celeba_128/results2/{output_image_index}-images.jpg"


path = 'data/mydata/test/Dark'
input_dark_files 	= os.listdir(path)

input_dark_index 	= 0
save_dir 			= 'output/Dark'		
save_index 			= 2					 



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

		# detect if Dark images are all done this condition only enter once 
		if input_dark_index == len(input_dark_files):
			sys.exit(0)

		# get current file name
		if i+1 < 10:
			tmp_str = '0'+str(i+1)
		else:
			tmp_str = str(i+1)
		cur_filename = os.path.join(save_dir,'_'+tmp_str+f'_0{save_index+1}.png')

		tar_filename = input_dark_files[input_dark_index]
		input_dark_index += 1
		tar_filename = os.path.join(save_dir, tar_filename.replace('jpg','png'))

		# rename file
		os.rename(cur_filename, tar_filename)

	# update output image path
	output_image_index += 1
	output_image_path = f"stargan_celeba_128/results2/{output_image_index}-images.jpg"