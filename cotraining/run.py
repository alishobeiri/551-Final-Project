from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.datasets import make_classification
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestClassifier
from classifiers import CoTrainingClassifier
import pickle 
import numpy as np


if __name__ == '__main__':
	pre_data = "../preprocessed/"
	data = "../data/"
	
	val = "citeseer"

	cl = 7
	if val == "citeseer":
		cl = 6

	tr_x = np.load(data + "ind." + val +'.allx', encoding="latin1").todense()
	t_x = np.load(data + "ind." + val +'.tx', encoding="latin1").todense()


	with open(data + 'ind.' + val + '.y', 'rb') as pickle_file:
		tr_y = pickle.load(pickle_file)

	with open(data + 'ind.' + val + '.ally', 'rb') as pickle_file:
		tr_all_y = pickle.load(pickle_file)
		
	with open(data + 'ind.' + val + '.ty', 'rb') as pickle_file:
		t_y = pickle.load(pickle_file)


	x = np.concatenate([tr_x, t_x])
	y = np.concatenate([tr_all_y, t_y])
	x = x[:y.shape[0]]
	y = np.argmax(y, axis=1)
	print("Max of y: ", max(y))
	print("y.shape:", y.shape)
	d_len = x.shape[0]
	# x, x_test, y, y_test = train_test_split(x, y, 
	# 										test_size=0.5)
	print("x: ", x.shape)
	print("y: ", y.shape)
	# print("x_test:", x_test.shape)
	# print("y_test:", y_test.shape)
	
	n_samples = x.shape[0]
	n_features = x.shape[1]

	indices = np.arange(n_samples)
	if val == "cora":
		n_labeled_points = 140
	elif val == "citeseer":
		n_labeled_points = 120

	# They take first n_labeled_points and unlabel the rest
	unlabeled_set = indices[n_labeled_points:]

	print("unlabeled set size: ", unlabeled_set.shape)
	print("indices.shape: ", indices.shape)
	y_train = np.copy(y)
	y_train[unlabeled_set] = -1

	N_SAMPLES = x.shape[0]
	N_FEATURES = x.shape[1]
	print("samples: ", N_SAMPLES)
	print("features: ", N_FEATURES)

	x1 = x[:, :n_features//2]
	x2 = x[:, n_features//2:]


	# Use following code testing parameters if you want
	
	
	cotrain_params = ParameterGrid({
		'k': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
		'u': [int(i*n_samples) for i in [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]],
		'p': [1, 2, 3, 10, 20, 30],
		'n': [1, 2, 3, 10, 20, 30]
	})

	svm_params = ParameterGrid({
			'loss': ['hinge','squared_hinge'],
            'C': [0.5, 2.0, 5.0, 50.0]
	})

	best_score = 0
	best_params = None
	print ('SVM CoTraining')
	for cotrain in cotrain_params:
		for svm in svm_params:
			svm_co_clf = CoTrainingClassifier(LinearSVC(**svm), **cotrain)
			y_pred = svm_co_clf.fit(x1, x2, y_train)
			score = accuracy_score(y, y_pred)
			if score > best_score:
				best_params = {
					"cotrain": cotrain,
					"svm": svm
				}
				best_score = score
				print(score)
	print("Best Params: ")
	print(best_params)

	print("Best score: ")
	print(best_score)
	


	
