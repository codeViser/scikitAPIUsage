from sklearn.datasets import load_digits
from matplotlib import pyplot as plt

digits = load_digits()

# fig = plt.figure(figsize=(3, 3))

# plt.imshow(digits['images'][66], cmap="gray", interpolation='none')

# plt.show()

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target)



from tensorflow.contrib import skflow
n_classes = len(set(y_train))
classifier = skflow.TensorFlowLinearClassifier(n_classes=n_classes)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

#accuracy calculated

# import numpy as np
# print(np.mean(y_test == y_pred))

#classification report

from sklearn import metrics
print(metrics.classification_report(y_true=y_test, y_pred=y_pred))