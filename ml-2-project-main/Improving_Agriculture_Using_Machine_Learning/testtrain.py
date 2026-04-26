import os
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.models import model_from_json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
'''
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle
'''
# Non-Binary Image Classification using Convolution Neural Networks
'''
path = 'PlantDiseaseDataset'
low_green = np.array([25, 52, 72])
high_green = np.array([102, 255, 255])

labels = []
X_train = []
Y_train = []

def getID(name):
    index = 0
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index        
    

for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        name = os.path.basename(root)
        if name not in labels:
            labels.append(name)
print(labels)

for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        name = os.path.basename(root)
        if 'Thumbs.db' not in directory[j]:
            img = cv2.imread(root+"/"+directory[j])
            img = cv2.resize(img, (64,64), interpolation=cv2.INTER_CUBIC)
            imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(imgHSV, low_green, high_green)
            mask = 255-mask
            res = cv2.bitwise_and(img, img, mask=mask)
            #res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
            lbl = getID(name)
            print(name+" "+root+"/"+directory[j]+" "+str(lbl))
            X_train.append(res.ravel())
            Y_train.append(lbl)
        
X_train = np.asarray(X_train)
Y_train = np.asarray(Y_train)
print(Y_train)
print(X_train.shape)
np.save('model/X.txt',X_train)
np.save('model/Y.txt',Y_train)
'''

X = np.load('model/X.txt.npy')
Y = np.load('model/Y.txt.npy')
print(X.shape)
X = X.astype('float32')
X = X/255

print(X.shape)
    
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]
Y = to_categorical(Y)


test = X[3].reshape(64,64,3)
cv2.imshow("aa",cv2.resize(test,(300,300)))
cv2.waitKey(0)

if os.path.exists('model/pca.txt'):
    with open('model/pca.txt', 'rb') as file:
        pca = pickle.load(file)
        X = pca.fit_transform(X)
    file.close()
else:
    pca = PCA(n_components = 1200)
    X = pca.fit_transform(X)
    with open('model/pca.txt', 'wb') as file:
        pickle.dump(pca, file)
    file.close()
    
print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
ann_model = Sequential()
ann_model.add(Dense(512, input_shape=(X_train.shape[1],)))
ann_model.add(Activation('relu'))
ann_model.add(Dropout(0.3))
ann_model.add(Dense(512))
ann_model.add(Activation('relu'))
ann_model.add(Dropout(0.3))
ann_model.add(Dense(y_train.shape[1]))
ann_model.add(Activation('softmax'))
ann_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#print(ann_model.summary())
hist = ann_model.fit(X, Y, batch_size=16,epochs=15, validation_data=(X_test, y_test))
ann_model.save_weights('model/model_weights.h5')            
model_json = ann_model.to_json()
with open("model/model.json", "w") as json_file:
    json_file.write(model_json)
json_file.close()
f = open('model/history.pckl', 'wb')
pickle.dump(hist.history, f)
f.close()


