'''
Code for Co-generation and Segmentation for Generalized Surgical Instrument Segmentation on Unlabelled Data (MICCAI 2021)

Tajwar Abrar Aleef - tajwaraleef@ece.ubc.ca & Megha Kalia - mkalia@ece.ubc.ca (equal contribution)
Robotics and Control Laboratory, University of British Columbia, Vancouver,
Canada
'''


import re
from glob import glob

# Reads all file paths for the given data. All paths are hardcoded here. Modify the paths as needed.
def data_path_loader(dataName):

	X, Y, X_val, Y_val = ([] for _ in range(4))

	if dataName == 'Endovis':

		# Endovis 2017 data (Allan, M., et al.: 2017 robotic instrument segmentation challenge. arXiv preprint arXiv:1902.06426 (2019))
		path = '../data/endovis/'
		X = sorted(glob(path + '/Training/images/*.png'), key=lambda f: int(re.sub('\D', '', f)))
		Y = sorted(glob(path + '/Training/segmentations/*.png'), key=lambda f: int(re.sub('\D', '', f)))
		X_val = sorted(glob(path + '/Validation/images/*.png'), key=lambda f: int(re.sub('\D', '', f)))
		Y_val = sorted(glob(path + '/Validation/segmentations/*.png'), key=lambda f: int(re.sub('\D', '', f)))

	elif dataName == 'UCL':

		# UCL (https://link.springer.com/chapter/10.1007%2F978-3-030-59716-0_67)
		#Video 1 well lit
		#video 6, well lit
		#video 8, well lit
		#Video 3, a bit dark
		#Video 2, very dark
		#Video 4, very dark
		#Video 5, very dark
		#video 7, super dark
		#video 9 well lit and kidney
		#video 10 a bit dark
		#video 11, very dark and blood
		#video 12 a bit dark and blood
		#video 13 well lit
		#video 14 a bit dark and blood

		path = '../data/UCL/'
		train_folders = ['Video_01', 'Video_02', 'Video_03', 'Video_04', 'Video_05','Video_06', 'Video_07', 'Video_08']

		val_folders = ['Video_09', 'Video_10']

		for indx in range(len(train_folders)):
			X = X + sorted(glob(path + train_folders[indx] + '/images/*.png'), key=lambda f: int(re.sub('\D', '', f)))
			Y = Y + sorted(glob(path + train_folders[indx] + '/ground_truth/*.png'), key=lambda f: int(re.sub('\D', '', f)))
		for indx in range(len(val_folders)):
			X_val = X_val + sorted(glob(path + val_folders[indx] + '/images/*.png'), key=lambda f: int(re.sub('\D', '', f)))
			Y_val = Y_val + sorted(glob(path + val_folders[indx] + '/ground_truth/*.png'), key=lambda f: int(re.sub('\D', '', f)))

	elif dataName == 'Surgery':

		# Surgical Data (Unlabelled)
		path = '../data/surgery/Training'
		X = sorted(glob(path + '/images/*.png'), key=lambda f: int(re.sub('\D', '', f)))
		#No Y labels for surgical data during training

		path = '../data/surgery/Validation'
		X_val = sorted(glob(path + '/images/*.png'), key=lambda f: int(re.sub('\D', '', f)))
		Y_val = sorted(glob(path + '/segmentations/*.png'), key=lambda f: int(re.sub('\D', '', f)))

	return [X, Y, X_val, Y_val]