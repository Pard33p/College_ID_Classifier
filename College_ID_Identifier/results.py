#IMPORT REQUIRED LIBRARIES
import numpy as np
from sklearn.metrics import confusion_matrix
import math
from keras.models import model_from_json
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

#GET SAVED MODEL
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)
# load weights into new model
classifier.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


#CHECK MODEL PERFORMANCE
n_test_examples = 583
test_datagen = ImageDataGenerator(rescale = 1./255)
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
print('Accuracy : '+str(accuracy))
print('Error : '+str(error))
#Confidence interval
const = 1.96
temp = const * math.sqrt((error * (1-error)) / n)
confidence_interval = (error - temp, error + temp)
print('Confidence interval for 95% is ['+str(confidence_interval[0])+', '+str(confidence_interval[1])+']')

#TEST YOUR OWN IMAGE

imageName = input('Enter name of image you want to check : ')
test_image = image.load_img(imageName, target_size=(50, 50))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
if result[0][0] == 1:
    print('Not ID card')
else:
    print('ID card')
