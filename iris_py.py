import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm, metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.naive_bayes import GaussianNB
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Load the iris dataset
iris = pd.read_csv('iris (1).csv')

# Check for null values
print(iris.isnull().sum())

# Drop the unnecessary column
iris = iris.drop('Unnamed: 0', axis=1)

# Count the number of samples for each species
n = len(iris[iris['Species'] == 'versicolor'])
print("No of Versicolor in Dataset:", n)

n1 = len(iris[iris['Species'] == 'setosa'])
print("No of setosa in dataset:", n1)

n2 = len(iris[iris['Species'] == 'virginica'])
print("No of virginica in dataset:", n2)

# Check for outliers using boxplots
plt.figure(1)
plt.boxplot([iris['Sepal.Length']])
plt.figure(2)
plt.boxplot([iris['Sepal.Width']])
plt.show()

# Plot histograms
iris.hist()
plt.figure(figsize=(10, 7))
plt.show()

# Split the data into train and test sets
train, test = train_test_split(iris, test_size=0.3)
print(train.shape)
print(test.shape)

train_X = train[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']]
train_y = train.Species

test_X = test[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']]
test_y = test.Species

# Logistic Regression
model1 = LogisticRegression()
model1.fit(train_X, train_y)
prediction = model1.predict(test_X)
print('Accuracy (Logistic Regression):', metrics.accuracy_score(prediction, test_y))

# Confusion matrix for Logistic Regression
confusion_mat = confusion_matrix(test_y, prediction)
print("Confusion matrix (Logistic Regression): \n", confusion_mat)

# Support Vector Machines
model2 = svm.SVC()
model2.fit(train_X, train_y)
pred_y = model2.predict(test_X)
print("Accuracy (SVM):", accuracy_score(test_y, pred_y))

# K-Nearest Neighbors
model3 = KNeighborsClassifier(n_neighbors=5)
model3.fit(train_X, train_y)
y_pred2 = model3.predict(test_X)
print("Accuracy (KNN):", accuracy_score(test_y, y_pred2))

# Naive Bayes
model4 = GaussianNB()
model4.fit(train_X, train_y)
y_pred3 = model4.predict(test_X)
print("Accuracy (Naive Bayes):", accuracy_score(test_y, y_pred3))

# Encode the target variable
encoder = LabelEncoder()
encoder.fit(train_y)
train_y_encoded = encoder.transform(train_y)
test_y_encoded = encoder.transform(test_y)

# One-hot encode the labels
train_y_encoded = to_categorical(train_y_encoded)
test_y_encoded = to_categorical(test_y_encoded)

# Build the TensorFlow model
model = Sequential()
model.add(Dense(12, input_dim=4, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(train_X, train_y_encoded, epochs=150, batch_size=10, verbose=1)

# Evaluate the model
_, accuracy = model.evaluate(test_X, test_y_encoded)
print('Accuracy (TensorFlow): %.2f' % (accuracy*100))

# Predict and evaluate
y_pred_tf = model.predict(test_X)
y_pred_tf_classes = np.argmax(y_pred_tf, axis=1)
test_y_classes = np.argmax(test_y_encoded, axis=1)
print("Accuracy Score (TensorFlow):", accuracy_score(test_y_classes, y_pred_tf_classes))

# Prepare the new results row
new_result = pd.DataFrame({
    'Model': ['TensorFlow Neural Network'],
    'Score': [accuracy]
})

# Concatenate the new result with the existing results DataFrame
results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Support Vector Machines', 'KNN', 'Naive Bayes'],
    'Score': [metrics.accuracy_score(prediction, test_y), accuracy_score(test_y, pred_y),
              accuracy_score(test_y, y_pred2), accuracy_score(test_y, y_pred3)]
})

results = pd.concat([results, new_result], ignore_index=True)

result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
print(result_df.head(9))
