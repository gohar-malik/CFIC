import os
import pandas as pd
import shutil

# anno gender file
df = pd.read_csv('images_val2014.csv', usecols=['id','bb_skin'])

# image dataset
path = 'val2014_faces'
files = os.listdir(path)

for file in files:
	print(file)
	key = file.split('_')[2].split('.')[0]
	# index not exist
	if df[df['id']==int(key)].index.values.shape == (0,):
		shutil.copy(os.path.join("val2014_faces",file), os.path.join("data/mydata/test/Undecided2",file))
	else:
		gender_index = df.iloc[df[df['id']==int(key)].index.values[0]]['bb_skin']
		if gender_index == 'Dark':
			shutil.copy(os.path.join("val2014_faces",file), os.path.join("data/mydata/test/Dark",file))
		elif gender_index == 'Light':
			shutil.copy(os.path.join("val2014_faces",file), os.path.join("data/mydata/test/Light",file))
		else:
			shutil.copy(os.path.join("val2014_faces",file), os.path.join("data/mydata/test/Undecided2",file))