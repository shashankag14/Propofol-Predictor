"""
@purpose : Class for RandomForest model to initiate, train and evaluate
@when : 09/01/22
"""
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt

class RandomForest(object):
	def __init__(self):
		# values as per C3AI, max_features = None : using all features when looking for best split
		self.rf = RandomForestRegressor(n_estimators = 100, max_depth = 15, max_features = None)

	# To plot the features as per the importance on a bar graph
	def get_feature_importance(self, input_features, model):
		f_i = list(zip(input_features, model.feature_importances_))
		f_i.sort(key=lambda x: x[1])
		plt.barh([x[0] for x in f_i], [x[1] for x in f_i])
		plt.show()

	# Training phase
	def train(self, train_input, train_target, input_features):
		self.rf.fit(train_input, train_target)
		self.get_feature_importance(input_features, self.rf)
		return self.rf

	# Eval phase on validation data - compute metrics (mdpe, mdape, r2) on this
	def eval(self, model, test_input, test_target):
		predictions = model.predict(test_input)
		errors = predictions - test_target
		mdpe = 100 * np.median(errors / test_target)        # median percentage error (MPE)
		mdape = 100 * np.median(abs(errors) / test_target)  # median absolute percentage error (MDAPE)
		r2 = r2_score(predictions, test_target)             # R2 score

		return mdpe, mdape, r2

	# Runs the training and evaluation phase
	def run(self, train_input, test_input, train_target, test_target, input_features):
		best_model = self.train(train_input, train_target, input_features)
		mdpe, mdape, r2 = self.eval(best_model, test_input, test_target)

		print('Random forest scores:')
		print('MDPE = {:0.2f}%, MDAPE = {:0.2f}%, R2 = {:0.2f}'.format(mdpe, mdape, r2))
