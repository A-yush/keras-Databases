import os,shutil
from keras import layers,models,optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

original_dataset_dir="/home/ayush/kaggleProjects/TensorFlow/train"
base_dir="/home/ayush/kaggleProjects/TensorFlow"
base_dir=os.path.join(base_dir,"catDogConv")
dir=[]
dir.append(base_dir)
#making directories
train_dir=os.path.join(base_dir,"train")
dir.append(train_dir)
validation_dir=os.path.join(base_dir,"valid")
dir.append(validation_dir)
test_dir=os.path.join(base_dir,"test")
dir.append(test_dir)

train_cats_dir=os.path.join(train_dir,"cats")
dir.append(train_cats_dir) 
train_dogs_dir = os.path.join(train_dir, 'dogs')
dir.append(train_dogs_dir) 
validation_cats_dir = os.path.join(validation_dir, 'cats')
dir.append(validation_cats_dir) 
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
dir.append(validation_dogs_dir) 
test_cats_dir = os.path.join(test_dir, 'cats')
dir.append(test_cats_dir) 
test_dogs_dir = os.path.join(test_dir, 'dogs')
dir.append(test_dogs_dir)

#copying cat and dog samples to resp directories
catNames=['cat.{}.jpg'.format(i) for i in range(1000)]   #copies 1000 pics to train dir
for catName in catNames:
	src=os.path.join(original_dataset_dir,catName)
	dst=os.path.join(train_cats_dir,catName)
	#shutil.copy(src,dst)

catNames=['cat.{}.jpg'.format(i) for i in range(1000,1500)]   #copies next 500 pica to valid dir
for catName in catNames:
	src=os.path.join(original_dataset_dir,catName)
	dst=os.path.join(validation_cats_dir,catName)
	#shutil.copy(src,dst)

catNames=['cat.{}.jpg'.format(i) for i in range(1500,2000)]  #copies next 500 to test dir
for catName in catNames:
	src=os.path.join(original_dataset_dir,catName)
	dst=os.path.join(test_cats_dir,catName)
	#shutil.copy(src,dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
	src = os.path.join(original_dataset_dir, fname)
	dst = os.path.join(train_dogs_dir, fname)
	#shutil.copyfile(src, dst)
fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
	src = os.path.join(original_dataset_dir, fname)
	dst = os.path.join(validation_dogs_dir, fname)
	#shutil.copyfile(src, dst) 
fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
	src = os.path.join(original_dataset_dir, fname)
	dst = os.path.join(test_dogs_dir, fname)
	#shutil.copyfile(src, dst)

print('total training cat images:', len(os.listdir(train_cats_dir)))
print('total training cat images:', len(os.listdir(train_dogs_dir)))

def make_dirs(list):
	for i in list:
		os.mkdir(i)

#making model
model=models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation="relu",input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(512,activation="relu"))
model.add(layers.Dense(1,activation="sigmoid"))
#print(model.summary())

#optimizing
model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])

#preprocessing
train_datagen=ImageDataGenerator(rescale=1./255)   #used for image preprocessing in the form of generators as 
test_datagen=ImageDataGenerator(rescale=1./255)    #dataset is very large.

train_generator=train_datagen.flow_from_directory(train_dir,target_size=(150,150),batch_size=20,class_mode="binary")
validation_generator=test_datagen.flow_from_directory(validation_dir,target_size=(150,150),batch_size=20,class_mode="binary")
#flow_from_diectory takes labels as subfolder names of directory

#fitting or training the model
history=model.fit_generator(train_generator,steps_per_epoch=100,epochs=30,validation_data=validation_generator,validation_steps=50)

#plotting the model
train_acc=history.history["acc"]
valid_acc=history.history["val_acc"]
train_loss=history.history["loss"]
valid_loss=history.history["val_loss"]
epochs=range(1,len(train_acc)+1)

plt.plot(epochs,train_acc,'bo',label="Train Accuracy")
plt.plot(epochs,valid_acc,'b',label="Valid Accuracy")
plt.plot(epochs,train_loss,'ro',label="Train Loss")
plt.plot(epochs,valid_loss,'r',label="Valid loss")
plt.title("Training and valid acc and loss")
plt.legend()
plt.show()

