import sys
import time
import numpy as np
import pickle
from sklearn import model_selection, svm, preprocessing
from sklearn.metrics import accuracy_score,confusion_matrix
from Fashion_MNIST_Loader.mnist_loader import MNIST
import matplotlib.pyplot as plt
from matplotlib import style
import cv2
style.use('ggplot')


# Save all the Print Statements in a Log file.
old_stdout = sys.stdout
log_file = open("SVM_summary.log","w")
sys.stdout = log_file

tic = time.time()

# Load MNIST Data
print('\nLoading MNIST Data...')
data = MNIST('./Fashion_MNIST_Loader/dataset/')

print('\nLoading Training Data...')
img_train, labels_train = data.load_training()
train_img = np.array(img_train)
train_labels = np.array(labels_train)

print('\nLoading Testing Data...')
img_test, labels_test = data.load_testing()
test_img = np.array(img_test)
test_labels = np.array(labels_test)


#Features
X = train_img

#Labels
y = train_labels

# Prepare Classifier Training and Testing Data
print('\nPreparing Classifier Training and Validation Data...')
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.1)


# Pickle the Classifier for Future Use
print('\nSVM Classifier with gamma = 0.1; Kernel = polynomial')
print('\nPickling the Classifier for Future Use...')
clf = svm.SVC(gamma=0.1, kernel='poly')
clf.fit(X_train,y_train)

with open('MNIST_SVM.pickle','wb') as f:
	pickle.dump(clf, f)

pickle_in = open('MNIST_SVM.pickle','rb')
clf = pickle.load(pickle_in)

print('\nCalculating Accuracy of trained Classifier...')
acc = clf.score(X_test,y_test)

print('\nMaking Predictions on Validation Data...')
y_pred = clf.predict(X_test)

print('\nCalculating Accuracy of Predictions...')
accuracy = accuracy_score(y_test, y_pred)

print('\nCreating Confusion Matrix...')
conf_mat = confusion_matrix(y_test,y_pred)

print('\nSVM Trained Classifier Accuracy: ',acc)
print('\nPredicted Values: ',y_pred)
print('\nAccuracy of Classifier on Validation Images: ',accuracy)
print('\nConfusion Matrix: \n',conf_mat)

# Plot Confusion Matrix Data as a Matrix
plt.matshow(conf_mat)
plt.title('Confusion Matrix for Validation Data')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
#plt.show()

test_img = cv2.imread('inp5.jpg')
test_img = test_img.astype(float)
print('\nMaking Predictions on Test Input Images...')
test_labels_pred = clf.predict(test_img)

#print('\nCalculating Accuracy of Trained Classifier on Test Data... ')
#acc = accuracy_score(test_labels,test_labels_pred)

#print('\n Creating Confusion Matrix for Test Data...')
#conf_mat_test = confusion_matrix(test_labels,test_labels_pred)

print('\nPredicted Labels for Test Images: ',test_labels_pred)
#print('\nAccuracy of Classifier on Test Images: ',acc)
#print('\nConfusion Matrix for Test Data: \n',conf_mat_test)

toc = time.time()

print('Total Time Taken: {0} ms'.format((toc-tic)*1000))

# Plot Confusion Matrix for Test Data
#plt.matshow(conf_mat_test)
#plt.title('Confusion Matrix for Test Data')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
#plt.show()

sys.stdout = old_stdout
log_file.close()

arr = ['T-Shirt/Top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']

# Show the Test Images with Original and Predicted Labels
a = np.random.randint(1,40,15)
for i in a:
	two_d = (np.reshape(test_img[i], (28, 28)) * 255).astype(np.uint8)
	plt.title('Predicted Label: {1}'.format(arr[test_labels_pred[i]]))
	plt.imshow(two_d, interpolation='nearest')
	plt.show()
#---------------------- EOC ---------------------#
