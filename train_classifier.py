# import pickle

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import numpy as np


# data_dict = pickle.load(open('./data.pickle', 'rb'))

# print(type(data_dict['data']))
# data = np.array(data_dict['data'])
# labels = np.array(data_dict['labels'])
# # X = np.array(train[features][0].tolist())

# x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# model = RandomForestClassifier()

# model.fit(x_train, y_train)

# y_predict = model.predict(x_test)

# score = accuracy_score(y_predict, y_test)

# print('{}% of samples were classified correctly !'.format(score * 100))

# f = open('model.p', 'wb')
# pickle.dump({'model': model}, f)
# f.close()

import pickle
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data_dict = pickle.load(open('./data.pickle', 'rb'))

# Assuming data_dict['data'] contains sequences of varying lengths
# You need to preprocess the data to ensure all sequences have the same length
# One common approach is padding sequences to the maximum length

# # Find the maximum sequence length
# max_length = max(len(seq) for seq in data_dict['data'])

# # Pad sequences to the maximum length
# data_padded = np.array([seq + [0] * (max_length - len(seq)) for seq in data_dict['data']])

# Convert data and labels to numpy arrays
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize and train the RandomForestClassifier model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Make predictions on the test set
y_predict = model.predict(x_test)

# Calculate accuracy
score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly!'.format(score * 100))

# Save the trained model to a file
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
