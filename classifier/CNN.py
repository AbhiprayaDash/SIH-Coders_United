# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 21:21:43 2018

@author: USER
"""

from keras.models import Sequential
from keras.layers import Dense,Convolution2D,MaxPooling2D,Flatten,BatchNormalization


classifier = Sequential()

classifier.add(Convolution2D(input_shape=(256,256,3),filters=96,kernel_size=(11,11),strides=4,activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Convolution2D(filters=256,kernel_size=(5,5),activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Convolution2D(filters=384,kernel_size=(3,3),activation = 'relu'))
classifier.add(Convolution2D(filters=384,kernel_size=(3,3),activation = 'relu'))
classifier.add(Convolution2D(filters=256,kernel_size=(3,3),activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))


classifier.add(Flatten())

classifier.add(Dense(output_dim=2048,activation='relu'))
classifier.add(Dense(output_dim=2048,activation='relu'))
classifier.add(Dense(output_dim=6,activation='softmax'))

classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

train_datagen = ImageDataGenerator(
                                   rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
                                                   'Training',
                                                    target_size=(256, 256),
                                                    batch_size=32,
                                                    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
                                                    'test',
                                                    target_size=(256, 256),
                                                    batch_size=32,
                                                    class_mode='categorical')

classifier.fit_generator(
        train_generator,
        steps_per_epoch=950,
        epochs=15,
        validation_data=validation_generator,
        validation_steps=2000)
from keras.preprocessing import image
import numpy as np
test_image = image.load_img('metal289.jpg',target_size = (256,256))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
result = classifier.predict(test_image)
train_generator.class_indices
if result[0][0]==1:
    prediction='chal'
elif result[0][0]==0:
    prediction='card'
elif result[0][0]==2:
    prediction='metal'
elif result[0][0]==3:
    prediction='paper'
elif result[0][0]==4:
    prediction='plastic'
elif result[0][0]==5:
    prediction='trash'
    

