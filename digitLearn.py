from sklearn.datasets import load_digits
from matplotlib import pyplot as plt

digits = load_digits()

# fig = plt.figure(figsize=(3, 3))

# plt.imshow(digits['images'][66], cmap="gray", interpolation='none')

# plt.show()

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target)

from sklearn import svm

classifier = svm.SVC(gamma=0.001)
classifier.fit(X_train, y_train)
predicted = classifier.predict(X_test)

#accuracy calculated

import numpy as np
print(np.mean(y_test == predicted))