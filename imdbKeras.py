from keras.datasets import imdb
from keras import models,layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
num_words=10000)

print(train_data[0])
print(train_labels[0])


#vectorize train data to 0's and 1's
def vectorizeComment(sequences,dimensions=10000):
	results=np.zeros((len(sequences),dimensions))
	for i,sequence in enumerate(sequences): #enumerate used to add counter to array
		results[i,sequence]=1
	return results

x_train=vectorizeComment(train_data)
#converting label as array and vectorize it
y_train=np.asarray(train_labels).astype('float32')

#x_test=vectorizeComment(test_data)
model=models.Sequential()
model.add(layers.Dense(16,kernel_regularizer=regularizers.l2(0.001),activation='relu',input_shape=(10000,))) #adding 3 layers
model.add(layers.Dense(16,kernel_regularizer=regularizers.l2(0.001),activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy']) #compiling and adding loss and activation fns

#validation
x_val=x_train[:10000]
partialX_train=x_train[10000:]

y_val=y_train[:10000]
partialY_train=y_train[10000:]

history = model.fit(partialX_train,partialY_train,epochs=20,batch_size=512,validation_data=(x_val, y_val))

#plotting result
history_dict = history.history  #contains the summary with .history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
accu_val=history_dict['acc']
epochs = range(1, len(accu_val) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss') #bo for blue dot
plt.plot(epochs, val_loss_values, 'b', label='Validation loss') #b for solid blue line
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


def backToWords(comment):
	word_index=imdb.get_word_index()  #getting words related to all integers 
	reverse_word_index=dict(
		[(value,key)for (key,value) in word_index.items()] #converting integer index to words key=integer val=word
	)
	decoded_review=' '.join(
		[reverse_word_index.get(i-3,'?')for i in comment]  #getting words related to that integer
	)

#print(backToWords(train_data[0]))
