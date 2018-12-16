import numpy as np
import random
import copy
import pdb

class CoTrainingClassifier(object):
	"""
	Parameters:
	clf - The classifier that will be used in the cotraining algorithm on the X1 feature set
		(Note a copy of clf will be used on the X2 feature set if clf2 is not specified).

	clf2 - (Optional) A different classifier type can be specified to be used on the X2 feature set
		 if desired.

	p - (Optional) The number of positive examples that will be 'labeled' by each classifier during each iteration
		The default is the is determined by the smallest integer ratio of positive to negative samples in L (from paper)

	n - (Optional) The number of negative examples that will be 'labeled' by each classifier during each iteration
		The default is the is determined by the smallest integer ratio of positive to negative samples in L (from paper)

	k - (Optional) The number of iterations
		The default is 30 (from paper)

	u - (Optional) The size of the pool of unlabeled samples from which the classifier can choose
		Default - 75 (from paper)

	c - (Optional) The number of classes in the multiclass problem
		Default - 7
	"""

	def __init__(self, clf, clf2=None, p=-1, n=-1, k=20, u=30, cl=7):
		self.clf1_ = clf
		
		#we will just use a copy of clf (the same kind of classifier) if clf2 is not specified
		if clf2 == None:
			self.clf2_ = copy.copy(clf)
		else:
			self.clf2_ = clf2

		#if they only specify one of n or p, through an exception
		if (p == -1 and n != -1) or (p != -1 and n == -1):
			raise ValueError('Current implementation supports either both p and n being specified, or neither')

		self.p_ = p
		self.n_ = n
		self.k_ = k
		self.u_ = u
		self.cl_ = cl
		random.seed()


	def fit(self, X1, X2, y):
		"""
		Description:
		fits the classifiers on the partially labeled data, y.

		Parameters:
		X1 - array-like (n_samples, n_features_1): first set of features for samples
		X2 - array-like (n_samples, n_features_2): second set of features for samples
		y - array-like (n_samples): labels for samples, -1 indicates unlabeled

		"""

		#we need y to be a numpy array so we can do more complex slicing
		y = np.asarray(y)

		#set the n and p parameters if we need to
		cl = 0
		while cl != self.cl_:
			print("cl: ", cl)
			if self.p_ == -1 and self.n_ == -1:
				num_pos = sum(1 for y_i in y if y_i == cl)
				num_neg = sum(1 for y_i in y if y_i != cl)
				
				n_p_ratio = num_neg / float(num_pos)
			
				if n_p_ratio > 1:
					self.p_ = 1
					self.n_ = round(self.p_*n_p_ratio)

				else:
					self.n_ = 1
					self.p_ = round(self.n_/n_p_ratio)
			assert(self.p_ > 0 and self.n_ > 0 and self.k_ > 0 and self.u_ > 0)			
			#the set of unlabeled samples
			U = [i for i, y_i in enumerate(y) if y_i == -1]
			print("Len U: ", len(U))
			print("Len y[y == -1]: ", y[y == -1].shape)
			#we randomize here, and then just take from the back so we don't have to sample every time
			random.shuffle(U)
			print(self.u_)
			#this is U' in paper
			U_ = U[-min(len(U), self.u_):]
			print("len(U_): ", len(U_))
			print("min(len(U), self.u_):", min(len(U), self.u_))
			#the samples that are initially labeled
			L = [i for i, y_i in enumerate(y) if y_i != -1]

			#remove the samples in U_ from U
			U = U[:-len(U_)]

			it = 0 #number of cotraining iterations we've done so far	
			count = 0
			#loop until we have assigned labels to everything in U or we hit our iteration break condition
			while it != self.k_ and U:
				it += 1
				self.clf1_.fit(X1[L], y[L])
				self.clf2_.fit(X2[L], y[L])

				y1 = self.clf1_.predict(X1[U_])
				y2 = self.clf2_.predict(X2[U_])

				n, p = [], []
				
				for i, (y1_i, y2_i) in enumerate(zip(y1, y2)):
					#we added all that we needed to for this iteration, so break
					if len(p) == 2 * self.p_ and len(n) == 2 * self.n_:
						break

					#update our newly 'labeled' samples.  Note that we are only 'labeling' a single sample
					#with each inner iteration.  We want to add 2p + 2n samples per outer iteration, but classifiers must agree

					if y1_i == y2_i and len(p) < self.p_:
						p.append((i, cl))

					if y2_i == y1_i != cl and len(n) < self.n_:
						n.append((i, y2_i))

				if len(p) != 0:
					# print("len(p): ", len(p))
					pass
				else:
					count += 1

				for x in p:
					y[U_[x[0]]] = x[1] 

				for x in n:
					y[U_[x[0]]] = x[1] 

				L.extend([U_[x[0]] for x in p])
				L.extend([U_[x[0]] for x in n])

				indices = p + [i[0] for i in n]
				U_temp = []
				for i, l_index in enumerate(U_):
					if i not in indices:
						U_temp.append(l_index)
				U_ = U_temp
				#add new elements to U_
				add_counter = 0 #number we have added from U to U_
				num_to_add = 2*len(p) + 2*len(n)
				while add_counter != num_to_add and U:
					add_counter += 1
					U_.append(U.pop())
			print("Total it: ", it)
			print("len(U): ", len(U))
			print("Total p zero: ", count)
			cl += 1
			print("\n")
		
		print(np.unique(y, return_counts=True))
		
		# let's fit our final model
		self.clf1_.fit(X1[L], y[L])
		self.clf2_.fit(X2[L], y[L])
		
		y_pred1 = self.clf1_.predict(X1)
		y_pred2 = self.clf2_.predict(X2)
		# When the two classifiers do not agree we randomly select one of their predictiosn
		for i, val in enumerate(y):
			if val == -1:
				y[i] = [y_pred1[i], y_pred2[i]][np.random.randint(0, 2)]

		print(np.unique(y, return_counts=True))

		return y

