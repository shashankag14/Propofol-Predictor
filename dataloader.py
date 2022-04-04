"""
@file : dataloader.py
@purpose : Fetch raw data, preprocess and split to train and test datasets
@note : get_data is the main method running this file
@when : 09/01/22
"""

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# import local files here
import utils

# macros used to refer to the columns of the raw data
COL_ID = 0
COL_TIME = 1
COL_CON_PLA = 2
COL_MG_MIN = 3
COL_WT = 4
COL_HT = 5
COL_AGE = 6
COL_SEX = 7

'''
Brief :     Wherever CON_PLA is not null in the raw data, there is a duplicate entry with the same time stamp.  
			This method copies the CON_PLA value to the previous row and removes this duplicate entry in order to avoid training biases
Inputs :    data : Raw data
Returns :   data : preprcessed data with duplicate entries removed 
'''
def process_duplicates(data):
	N = data.shape[0]                                           # total number of entries in raw data
	data = np.c_[data, np.zeros(N)]                             # add extra column to flag the duplicate rows which need to be removed

	for row in range(N - 1):
		# if ID and TIME are identical in two consecutive rows, its a duplicate entry
		if data[row][COL_ID] == data[row + 1][COL_ID] and data[row][COL_TIME] == data[row + 1][COL_TIME]:
			data[row][COL_CON_PLA] = data[row + 1][COL_CON_PLA] # copy the CON_PLA to the former duplicate row
			data[row + 1][-1] = 1                               # flag the later duplicate row as it needs to be removed

	data = np.delete(data, np.where(data[:, -1] == 1), axis=0)  # delete all thw rows with flag=1 in the last row
	data = np.delete(data, -1, axis=1)                          # delete the newly added flag column
	return data

'''
Brief :     This method fetches the last CON_PLA value and populates it to a new column besides the current CON_PLA value
			Note : It is done on per-patient basis !
Inputs :    data : Raw data
Returns :   data : preprocessed data with additional column containing previous CON_PLA 
			not_null_indices.tolist() : Indices of raw data where CON_PLA is not null
'''
def fill_prev_conpla(data):
	not_null_indices = np.where(~np.isnan(data[:, COL_CON_PLA]))[0]   # fetch row indices with non null CON_PLA values
	data = np.c_[data, np.zeros(data.shape[0])]                       # add a new column with all entries as 0

	for idx in range(1, len(not_null_indices)):
		prev = not_null_indices[idx-1]
		current = not_null_indices[idx]
		if data[current, COL_CON_PLA] == 0:                            # if current non null CON_PLA = 0, PREV_CON_PLA = 0
			data[current, 8] = 0
		else :
			data[current, 8] = data[prev, COL_CON_PLA]
	return data, not_null_indices.tolist()

'''
Brief   :   This method extracts out mg_min between two consecutive CON_PLA so as to use it as rows instead of column values 
			in the final preproc data. 
Input   :   propofol_array : raw data in numpy array form
			not_null_indices : List of indices in raw data array where CON_PLA is not null	
			patient_first_row : List of first row indices of each patient		
Returns :   all_mgmin_array : Nested list of mg_min between two consecutive CON_PLA
			max_gap : Maximum time lag between two consecutive measured CON_PLA			
'''
def extract_mgmin_per_conpla(propofol_array, not_null_indices, patient_first_row):
	all_mgmin = []
	max_ids_list = []
	max_gap = 0
	# begin_dx and end_idx - non null CON_PLA indices between which mg_ming needs to be extracted
	for i in range(0, len(not_null_indices) - 1):
		end_idx = not_null_indices[i + 1]
		# if CON_PLA = 0, start mg_min extraction from the same row onwards
		if propofol_array[not_null_indices[i], COL_CON_PLA] == 0:
			begin_idx = not_null_indices[i]
		# if CON_PLA is not zero, current row mg_min is for prev batch. Thus, start mg_min extraction from next row onwards
		else:
			begin_idx = not_null_indices[i] + 1

		# Skip the last mg_min for which we do not have CON_PLA
		if end_idx not in patient_first_row:
			between_mgmin = propofol_array[begin_idx:end_idx + 1, COL_MG_MIN].tolist()

			if len(between_mgmin) > max_gap:
				max_gap = len(between_mgmin)
				max_idx = end_idx
				max_ids_list.append(end_idx)

			all_mgmin.append(between_mgmin)
	print("Max gap between two measured CON_PLA: ", max_gap, " at index:", max_idx)
	# print("Gap more than 30minutes at indices : ", max_ids_list)

	'''
	Each row in all_mgmin represents the mg_min for measured CON_PLA.
	Num of mg_min for each such CON_PLA might not be the same
	Max num of mg_min for a measured CON_PLA is 68 (max_gap)
	Balance each row of all_mgmin with same num of mg_min entries by appending 0s
	'''
	for row in all_mgmin:
		while len(row) < max_gap:
			row.append(0)
	all_mgmin_array = np.array(all_mgmin)

	return all_mgmin_array, max_gap

'''
Brief   : Main method to start the preprocessing of the raw data
			1. Remove duplicate entries (rows) where CON_PLA is not null
			2. Add new column - PREV_CON_PLA 
			3. Extract mg_min between two consecutive CON_PLA and transpose the data
			4. Perform running cummulative sum of mg_min
Input   : Path to the raw data
Returns : Preprocessed data - trainingdata.xlsx
'''
def preprocess_data(path):
	# float precision added bcz pandas was changing the decimal values
	propofol_df = pd.read_csv(path, float_precision='round_trip')

	# Convert from dataframe to 2d array
	propofol_array = np.asarray(propofol_df)
	propofol_array = process_duplicates(propofol_array)     # remove the duplicate row entries when CON_PLA is measured

	patient_ids = np.unique(propofol_array[:,0]).astype(int)    # fetch the patient ids
	patient_first_row = []                                      # first row of each patient in the data
	patient_rows = []
	for id in patient_ids:
		patient_first_row.append(np.where(propofol_array[:, 0] == id)[0][0].tolist())
		patient_rows.append(np.where(propofol_array[:, 0] == id)[0].tolist())

	# Populate 0 in the first row of each patient. Helps in using 0 as the PREV_CON_PLA for the first CON_PLA
	for i in patient_first_row:
		propofol_array[i][COL_CON_PLA] = 0

	# Add a new column to the data, fill PREV_CON_PLA in the rows with non null CON_PLA, remaining rows with value 0
	propofol_array, not_null_indices = fill_prev_conpla(propofol_array)
	input_data_df = pd.DataFrame(propofol_array)

	# Filter out rows with non-null CON_PLA
	input_data_df = input_data_df[~input_data_df[COL_CON_PLA].isna()]
	# But do not consider rows with CON_PLA=0 as it was manually added for preproc
	input_data_df = input_data_df[input_data_df[COL_CON_PLA] != 0]

	# Drop columns - time [1] and mg_min [3]
	input_data_df = input_data_df.drop(input_data_df.columns[[COL_TIME,COL_MG_MIN]], 1).reset_index(drop=True)
	# input_data_df.to_excel(utils.data_path + 'input_data_df.xlsx')      # kept to cross verify, can be removed

	# Extract mg_min between two consecutive measured CON_PLA and transpose the data
	all_mgmin_array, max_num_mgmin = extract_mgmin_per_conpla(propofol_array, not_null_indices, patient_first_row)
	all_mgmin_df = pd.DataFrame(all_mgmin_array).reset_index(drop=True) # just to keep the order of the columns intact

	# Perform running cummulative sum of mg_min along rows (axis=1)
	all_mgmin_df = all_mgmin_df.cumsum(axis=1)
	# all_mgmin_df.to_excel(utils.data_path + 'all_mgmin_df.xlsx')          # kept to cross verify, can be removed

	trainingdata = pd.concat([input_data_df, all_mgmin_df], axis=1)

	original_column_names = ["ID", "CON_PLA", "WT", "HT", "AGE", "SEX", "PREV_CON_PLA"]
	time_column_names = ["Time" + str(num) for num in range(max_num_mgmin)] # time0 to time[xyz]
	trainingdata.columns = original_column_names + time_column_names
	trainingdata.to_excel(utils.data_path + 'trainingdata.xlsx')            # FINAL TRAINING DATA FILE

	return np.asarray(trainingdata), trainingdata.columns

'''
Brief   : Method to call the preprocessing method and then split the preprocessed data into train+test
Input   : Path to the raw data
Returns : Train and test input and targets, features of training data (ht, wt, age, etc)
'''
def get_data(path):
	# preprocess the data before training
	trainingdata, header = preprocess_data(path)

	# CON_PLA is the target
	target = trainingdata[:, 1]
	print("Target (CON_PLA) shape : ", target.shape)

	# WT, HT, SEX, AGE, PREV_CON_PLA and Time0-Time30
	input = np.delete(trainingdata, np.s_[0,1], axis=1)
	input_features = header[2:]
	print("Input shape : ", input.shape)

	# Split (input,target) into train/test - 90%, 10% resp.
	train_input, test_input, train_target, test_target = train_test_split(input,
	                                                                      target,
	                                                                      test_size = 0.1,
	                                                                      random_state = 42)
	print("Train data size : {}\nTest data size : {}".format(train_input.shape[0], test_input.shape[0]))

	return train_input, test_input, train_target, test_target, input_features
