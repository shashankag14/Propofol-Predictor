"""
@purpose : Class for NeuralNetwork model to initiate, train and evaluate
@when : 09/01/22
"""

import torch.nn

class NeuralNetwork(object):
	def __init__(self):
		pass

	def train(self, input_train, target_train):
		best_model = 0
		return best_model

	def eval(self, model, input_test, target_test):
		pass


	def run(self, input_train, target_train, input_test, target_test, input_features):
		best_model = self.train(input_train, target_train)
		self.eval(input_test, target_test)