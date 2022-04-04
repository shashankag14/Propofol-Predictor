"""
@purpose : Main file to initiate all the tasks and train the model. Please run this file in order to compile and excute the code
@when : 09/01/22
NOTE : Refer to the description of utils.py before running this file
"""

# import local files here
import dataloader
from model.randomforest import RandomForest
from model.neuralnet import NeuralNetwork
import utils

import warnings
warnings.filterwarnings("ignore")

# Fetch hyper-params from utils.py
args = utils.parser.parse_args()

# Fetch train and test data from dataloader
train_input, test_input, train_target, test_target, input_features = dataloader.get_data(args.data_path + utils.raw_data)

# Choose the model as per user input - Randomforest / NN ?
print("Running " + args.model)
if args.model=="rf":
	model_obj = RandomForest()
elif args.model=="nn":
	model_obj = NeuralNetwork()
else :
	print("Model choice unavailable.")

#run the preferred model on train and eval data
model_obj.run(train_input, test_input, train_target, test_target, input_features)
