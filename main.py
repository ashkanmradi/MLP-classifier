from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
# %matplotlib inline

iris_data = datasets.load_iris()
x_iris = iris_data.data 
y_iris = iris_data.target
iris_clf = sklearn.neural_network.MLPClassifier(hidden_layer_sizes = (10,), learning_rate = 'invscaling', solver='lbfgs',random_state=5, shuffle=True)
iris_cv_scores = cross_val_score(iris_clf, x_iris, y_iris, cv = 5)
print('cross-validation scores(5-fold):', iris_cv_scores)
print('mean cross-validation score(5-fold): {:.2f}'.format(np.mean(iris_cv_scores)))


glass_data = np.loadtxt('glass.data', delimiter=",")
x_glass = glass_data[:, :10]
y_glass = glass_data[:, 10]
glass_clf = sklearn.neural_network.MLPClassifier(hidden_layer_sizes = (10,), learning_rate = 'invscaling', solver='lbfgs',random_state=5, shuffle=True)
glass_cv_scores = cross_val_score(glass_clf, x_glass, y_glass, cv = 5)
print('cross-validation scores(5-fold):', glass_cv_scores)
print('mean cross-validation score(5-fold): {:.2f}'.format(np.mean(glass_cv_scores)))


plt.figure(figsize=(7, 5))
plt.grid(True)
for num_hidden_units in range(1,101):
    iris_clf = sklearn.neural_network.MLPClassifier(hidden_layer_sizes = (num_hidden_units,),
                                                    learning_rate = 'invscaling', solver='lbfgs',random_state=5, shuffle=True)
    iris_cv_scores = cross_val_score(iris_clf, x_iris, y_iris, cv = 5)
    print('mean cross-validation score(5-fold) having',num_hidden_units,'neurons in the hidden layer: {:.2f}'.format(np.mean(iris_cv_scores)))
    plt.scatter(num_hidden_units, np.mean(iris_cv_scores))
    plt.xlabel('number of hidden units in the hidden layer for iris-dataset')
    plt.ylabel('mean of the 5-fold cross-validation score')

plt.show()

plt.figure(figsize=(7, 5))
plt.grid(True)
for num_hidden_units in range(1,101):
    glass_clf = sklearn.neural_network.MLPClassifier(hidden_layer_sizes = (num_hidden_units,), 
                                                     learning_rate = 'invscaling', solver='lbfgs',random_state=5,shuffle=True)
    glass_cv_scores = cross_val_score(glass_clf, x_glass, y_glass, cv = 5)
    print('mean cross-validation score(5-fold) having',num_hidden_units,'neurons in the hidden layer: {:.2f}'.format(np.mean(glass_cv_scores)))
    plt.scatter(num_hidden_units, np.mean(glass_cv_scores))
    plt.xlabel('number of hidden units in the hidden layer for glass-dataset')
    plt.ylabel('mean of the 5-fold cross-validation score')

plt.show()

x_train_iris, x_test_iris, y_train_iris, y_test_iris = train_test_split(x_iris, y_iris, test_size = 0.3, random_state = 0)
plt.figure(figsize=(7, 5))
plt.title('Train and Test Accuracy for Iris Dataset')
plt.grid(True)
for num_hidden_units in range(1,502,10):
    iris_clf_2 = sklearn.neural_network.MLPClassifier(hidden_layer_sizes = (num_hidden_units,),
                                                      learning_rate = 'invscaling', solver='lbfgs',random_state=5,shuffle=True)
    iris_clf_2.fit(x_train_iris, y_train_iris)
    iris_predictions = iris_clf_2.predict(x_test_iris)
    print('iris accuracy on training set, having',num_hidden_units,'nuerons in hidden layer: ',accuracy_score(y_train_iris,
                                                                                                              iris_clf_2.predict(x_train_iris)))
    print('iris accuracy on test set, having',num_hidden_units,'nuerons in hidden layer: ',accuracy_score(y_test_iris,
                                                                                                              iris_clf_2.predict(x_test_iris)))
    plt.scatter(num_hidden_units, accuracy_score(y_train_iris, iris_clf_2.predict(x_train_iris)), c = 'red', marker = '+', s = 80)
    plt.scatter(num_hidden_units, accuracy_score(y_test_iris, iris_clf_2.predict(x_test_iris)), c = 'blue', marker = '*', s = 80)
    plt.xlabel('number of hidden units in the hidden layer')
    plt.ylabel('Accuracy Score')
    plt.ylim([0,1.5])
plt.show()

x_train_glass, x_test_glass, y_train_glass, y_test_glass = train_test_split(x_glass, y_glass, test_size = 0.3, random_state = 0)
plt.figure(figsize=(7, 5))
plt.title('Train and Test Accuracy for Glass Dataset')
plt.grid(True)
for num_hidden_units,num_epochs in zip(range(1,502,10),range(500,1502,10)):
    glass_clf_2 = sklearn.neural_network.MLPClassifier(hidden_layer_sizes = (num_hidden_units,),
                                                       learning_rate = 'invscaling', solver = 'lbfgs',random_state=5,shuffle=True)
    glass_clf_2.fit(x_train_glass, y_train_glass)
    glass_predictions = glass_clf_2.predict(x_test_glass)
    print('glass accuracy on training set, having',num_hidden_units,'nuerons in hidden layer: ',accuracy_score(y_train_glass,
                                                                                                                glass_clf_2.predict(x_train_glass)))
    print('glass accuracy on test set, having',num_hidden_units,'nuerons in hidden layer: ',accuracy_score(y_test_glass,
                                                                                                              glass_clf_2.predict(x_test_glass)))
    plt.scatter(num_hidden_units, accuracy_score(y_train_glass, glass_clf_2.predict(x_train_glass)), c = 'red', marker = '+', s = 80)
    plt.scatter(num_hidden_units, accuracy_score(y_test_glass, glass_clf_2.predict(x_test_glass)), c = 'blue', marker = '*', s = 80)
    plt.xlabel('number of hidden units in the hidden layer')
    plt.ylabel('Accuracy Score')
    plt.ylim([0,1.5])

plt.show()

iris_clf = sklearn.neural_network.MLPClassifier(hidden_layer_sizes = (10,10,), learning_rate = 'invscaling', solver='lbfgs',random_state=5, shuffle=True)
iris_cv_scores = cross_val_score(iris_clf, x_iris, y_iris, cv = 5)
print('cross-validation scores(5-fold):', iris_cv_scores)
print('mean cross-validation score(5-fold): {:.2f}'.format(np.mean(iris_cv_scores)))


glass_clf = sklearn.neural_network.MLPClassifier(hidden_layer_sizes = (10,10,),learning_rate = 'invscaling', solver='lbfgs',random_state=5, shuffle=True)
glass_cv_scores = cross_val_score(glass_clf, x_glass, y_glass, cv = 5)
print('cross-validation scores(5-fold):', glass_cv_scores)
print('mean cross-validation score(5-fold): {:.2f}'.format(np.mean(glass_cv_scores)))