#Convolutional Neural Network for identification of college ID card
'''
Following Parts are to be executed in order:
    Part 1
    Part 3
    Part 7
    Part 5
    Part 8(for testing one picture at a time)

'''


#Part 1 - Import required libraries 
import numpy as np
from sklearn.metrics import confusion_matrix
import math
from keras.models import model_from_json
from keras.preprocessing import image

# Part 2 - Building CNN

#Importing keras libraries and packages
from keras.models import Sequential
from keras.layers import (Convolution2D, MaxPooling2D, Flatten, Dense)

# Initializing the CNN
classifier = Sequential()

# Convolution
classifier.add(Convolution2D(32,3,3, input_shape=(50,50,3), activation = 'relu'))

# Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32,3,3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Flatten
classifier.add(Flatten())

#  Full Connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

#  Compiling CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Part 3 - Specify following
n_training_examples = 2569
n_test_examples = 583


# Part 4 - Fitting the CNN to images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255, 
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (50,50),
                                                 batch_size = 32, 
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                                 target_size = (50,50),
                                                 batch_size = 32, 
                                                 class_mode = 'binary')


classifier.fit_generator(training_set, 
                         samples_per_epoch = n_training_examples,
                         nb_epoch = 10,
                         validation_data = test_set,
                         nb_val_samples = n_test_examples)

# Part 5 - Calculate confusion matrix, accuracy, error and confidence interval
n_test_examples = 583
itr = test_datagen.flow_from_directory('dataset/test_set',
                                                 target_size = (50,50),
                                                 batch_size = n_test_examples, 
                                                 class_mode = 'binary')

X_test, y_test = itr.next()
y_pred = classifier.predict(X_test)
y_pred = np.rint(y_pred)

n = len(y_test)

cm = confusion_matrix(y_test, y_pred)

(tn, fp), (fn, tp) = cm

accuracy = (tp+tn) / n
error    = (fn+fp) / n

#Confidence interval
const = 1.96
temp = const * math.sqrt((error * (1-error)) / n)
confidence_interval = (error - temp, error + temp)


#Part 6 - SAVE THE CLASSIFIER FOR LATER USE

# serialize model to JSON
model_json = classifier.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("model.h5")
print("Saved model to disk")
 
# later...
 
# Part 7 - load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)
# load weights into new model
classifier.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# Part 8 - Insert your test image here
test_image = image.load_img('test.jpg', target_size=(50, 50))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    print('Not ID card')
else:
    print('ID card')
 



# MY Results
'''
Epoch 7/10
80/80 [==============================] - 20s 253ms/step - loss: 0.0360 - acc: 0.9887 - val_loss: 0.0609 - val_acc: 0.9833
Epoch 8/10
80/80 [==============================] - 20s 254ms/step - loss: 0.0216 - acc: 0.9922 - val_loss: 0.3570 - val_acc: 0.8666
Epoch 9/10
80/80 [==============================] - 20s 253ms/step - loss: 0.0283 - acc: 0.9896 - val_loss: 0.0316 - val_acc: 0.9949
Epoch 10/10
80/80 [==============================] - 20s 254ms/step - loss: 0.0338 - acc: 0.9895 - val_loss: 0.0601 - val_acc: 0.9801
Out[12]: <keras.callbacks.History at 0x7fbdc76040b8>

Accuracy : 98.01 %
Error    : 1.99 %


'''



