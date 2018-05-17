from keras.datasets import reuters
from keras.utils.np_utils import to_categorical
from keras import models,layers
import numpy as np
import matplotlib.pyplot as plt


(train_data,train_labels),(test_data,test_labels) = reuters.load_data(num_words=10000)


def vectorSeq(sequences,dimensions=10000):
	results=np.zeros((len(sequences),dimensions))
	for i,sequence in enumerate(sequences):
		results[i,sequence]=1
	return results

def decodeToWords(sequence):
	wordIndex=reuters.get_word_index()
	revIndex=dict(
		[(value,key)for (key,value) in wordIndex.items()])
	decWords=" ".join([revIndex.get(i-3,'?') for i in sequence])
	return decWords

x_train=vectorSeq(train_data)
x_test=vectorSeq(test_data)

y_train=to_categorical(train_labels)  #one hot encoding of labels
y_test=to_categorical(test_labels)

#setting network
model=models.Sequential()
model.add(layers.Dense(64,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(46,activation='softmax'))
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

#validation
x_val=x_train[:1000]
rem_x_train=x_train[1000:]

y_val=y_train[:1000]
rem_y_train=y_train[1000:]

#fitting
history=model.fit(rem_x_train,rem_y_train,epochs=20,batch_size=512,validation_data=(x_val,y_val))

#plotting
loss=history.history['loss']
val_loss=history.history['val_loss']
acc=history.history['acc']
val_acc=history.history['val_acc']
epochs=range(1,len(loss)+1)
plt.plot(epochs,loss,'bo',label="Training loss")
plt.plot(epochs,val_loss,'b',label="Validation Loss")
plt.plot(epochs,acc,'ro',label="Training Accuracy")
plt.plot(epochs,val_acc,'r',label="Validation Accuracy")
plt.xlabel("epochs")
plt.ylabel("loss and accuracy")
plt.legend()
plt.show()