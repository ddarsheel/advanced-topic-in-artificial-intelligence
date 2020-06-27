# -*- coding: utf-8 -*-
"""
Suprith kangokar        N10124021
Darsheel Deshpande      N10287213
Karthik  kadadevarmath  N10281797 

INSTRUCTION TO RUN CODE

1) Install tensorflow
1) Install keras
3) run code

"""


from keras.datasets import fashion_mnist
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.optimizers import RMSprop,SGD
from keras import backend as K
import numpy as np
import random
import keras
from keras.datasets import mnist
from keras.models import Model

#=====================================#
#                                     #
#      All functions declaration      #
#                                     #
#=====================================#

'''This function will return the loaded fashion mnist dataset '''
def load_fashionmnist_data():
  
  (x1,y1),(x2,y2) = fashion_mnist.load_data()
  dataset = np.concatenate((x1,x2),axis = 0)
  dataset_labels = np.concatenate((y1,y2),axis = 0)
  dataset = np.expand_dims(dataset,axis=3)
  return dataset,dataset_labels

'''Function to normalize values between 0 and 1
PArameters: data =>  dataset to normalize'''
def normalize_dataset(data):
  data = data.astype('float32')
  data /= 255
  return data



'''This function will take three parameteres and returns indexes 
Parameters: dataset_labels => '''
def find_indexes(dataset_labels,labels_list):
  
    indexes = [np.where(dataset_labels == i)[0] for i in labels_list]
    return indexes

'''Function to split data and make pairs'''
def split_and_make_pairs(data,labels,trainlabels,testlabels):
    '''First we will calculate indexes of train and all test data labels'''
    #finding trainlabels indexes
    train_indexes = find_indexes(labels,trainlabels)
    
    #finding test1 labels indexes which is 20% of train data
    min_len = len(train_indexes[0])
    threshold = int(min_len*0.2)
    
    test1_indexes = []
    for i in range(len(train_indexes)):
        
        test1_indexes.append(train_indexes[i][-threshold:])
        train_indexes[i] = train_indexes[i][:-threshold]
    
    #finding test2 labels indexes ("dress", "sneaker", "bag", "shirt")
    test2_indexes = find_indexes(labels,testlabels)
    
    #finding test1_union_test2 labels indexes 
    test1_union_test2_indexes = test2_indexes.copy()

    for i in range(len(test1_indexes)):
      test1_union_test2_indexes.append(test1_indexes[i])
    
    '''Creating pairs of data'''
    train_data_pairs , train_data_labels = makepairs_images(data, train_indexes)
    test1_data_pairs , test1_data_labels = makepairs_images(data, test1_indexes)
    test2_data_pairs , test2_data_labels = makepairs_images(data, test2_indexes)
    test1utest2_data_pairs , test1utest2_data_labels = makepairs_images(data, test1_union_test2_indexes)
    
    train_data_pairs = normalize_dataset(train_data_pairs)
    test1_data_pairs = normalize_dataset(test1_data_pairs)
    test2_data_pairs = normalize_dataset(test2_data_pairs)
    test1utest2_data_pairs = normalize_dataset(test1utest2_data_pairs)
    
    return train_data_pairs , train_data_labels,test1_data_pairs , test1_data_labels,test2_data_pairs , test2_data_labels,test1utest2_data_pairs , test1utest2_data_labels
    
'''This function takes data and labels and form random pairs from same category images to different category images
Paramaters: data => mean dataset that we will pass to make pairs || labels_places => is a 2d list containng list of indexes of each label in dataset'''
def makepairs_images(data, labels_places):
  
    class_num = int(len(labels_places))
    
    data_pairs = []
    data_labels = []
    n = min([len(labels_places[d]) for d in range(class_num)]) - 1
    for d in range(class_num):
        for i in range(n):
            z1, z2 = labels_places[d][i], labels_places[d][i + 1]
            data_pairs += [[data[z1], data[z2]]]
            inc = random.randrange(1, class_num)
            dn = (d + inc) % class_num
            z1, z2 = labels_places[d][i], labels_places[dn][i]
            data_pairs += [[data[z1], data[z2]]]
            data_labels += [1, 0]
    return np.array(data_pairs), np.array(data_labels)
  
  
'''This function is to pass in siamese model metrics to calculate accuracy during training'''
def model_accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))
  
  
'''This Is custom loss function which will calculate the loss of similarity and difference
#Contrastive loss function
Parameters: y_true=> ground truth labels , y_pred => Model prediction scores'''
def custom_loss_function(y_true, y_pred):
    
    limit = 1
    #squaring model prediction
    squarepred = K.square(y_pred)
    #calculating loss on base of difference between o and predicted value
    marginsquare = K.square(K.maximum(limit - y_pred, 0))
    return K.mean(y_true * squarepred + (1 - y_true) * marginsquare)
  
  
'''This function will return the L2 distance which is euclidean distance between 2 feature vectors of same size
Parameters: features => Contains 2 features vector obtained from Siamese network'''
def L2_distance(features): 
    x1_features, x2_features = features
    #finding sum of differences of two vectors
    sum1 = K.sum(K.square(x1_features - x2_features), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum1, K.epsilon()))

  

'''This function is to set the output shape of model'''
def model_shape(size): 
    size1, size2 = size
    return (size1[0], 1)


'''Function to declare siamese model architecture.Returns model object
Parameters: shape1 is image shape (28*28*1)''' 
def network_architecture(shape1):
    input = Input(shape=shape1)
    layer1 = keras.layers.Conv2D(16, kernel_size=(3, 3))(input)
    layer2 = keras.layers.Flatten()(layer1)
    layer3 = Dropout(0.4)(layer2)
    layer4 = Dense(256, activation='relu')(layer3)
    layer5 = Dropout(0.6)(layer4)
    layer6 = Dense(128, activation='relu')(layer5)
    return Model(input, layer6)
  
'''This function will build the siamese complete model from common architecture and return a compiled model'''
def Build_siamese_model(train_data,learningRate=0.00001):
    input_shape = train_data.shape[2:]

    #Creating network architecture
    shared_network = network_architecture(input_shape) #input_shape avriable is input shape of the model 28*28*1 .

    #Declaring keras Input objects 
    input_1 = Input(shape=input_shape)
    input_2 = Input(shape=input_shape)

    #Declaring two objects of shared network
    shared1 = shared_network(input_1)
    shared2 = shared_network(input_2)

    #Concatenating two base model instances to make its output to one model
    distance = Lambda(L2_distance,
                      output_shape=model_shape)([shared1,shared2])

    #Declaring one final model
    siamese_model = Model([input_1, input_2], distance)

    # Compiling model with SGD optimizer
    
    rms = RMSprop(lr=learningRate, rho=0.9)
    sgd = SGD(lr=learningRate, decay=1e-6, momentum=0.9, nesterov=True)
    siamese_model.compile(loss=custom_loss_function, optimizer=sgd, metrics=[model_accuracy])
    return siamese_model
    
    
'''Function To train model
Parameters: model=> model to train,train_data=> train data ,train_labels=> training data labels(0,1),
val_data=> Validation to test model,val_labels => Validation data labels, batchsize => batchsize to train model on batches ,epochs => Total epochs to train model on'''
def train(model,train_data,train_labels,val_data,val_labels,batchsize,epochs):
    loss_acc_history = model.fit([train_data[:, 0], train_data[:, 1]], train_labels,
                        batch_size=batchsize,
                        epochs=epochs,
                        validation_data=([val_data[:, 0], val_data[:, 1]], val_labels))
    
    return model,loss_acc_history
  
  
'''Function to Plot training and validation loss over epochs
Parameters: train_loss => training loss during training on train data, val_loss => validation loss on validation data'''
def plt_loss_curve(train_loss,val_loss):
    import matplotlib.pyplot as plt
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    
  
'''Function to Plot training and validation accuracy over epochs
Parameters: train_acc => training accuracy during training on train data, val_acc => validation accuracy on validation data'''
def plt_acc_curve(train_acc,val_acc):
    import matplotlib.pyplot as plt
    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.title('model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

'''This function calculates accuracy of model predictiong with respect to ground truth labels
Parameters: y_true=> ground truth labels , y_pred => Model prediction scores'''
def calculate_accuracy(y_true, y_pred): 
  
    prediction = y_pred.ravel() < 0.5
    return np.mean(prediction == y_true)

'''Function to print accuracy of different testing sets
Parameters: siamese_model => model to test, data=> data to undergo, labels => Ground truth labels of data , title => title of print statement'''
def print_accuracy(siamese_model,data,labels,title):
  
  prediction = siamese_model.predict([data[:, 0], data[:, 1]])
  acc = calculate_accuracy(labels, prediction)
  print(str(title)+': %0.2f%%' % (100 * acc))
  return prediction

'''Function to plot image pairs with their ground truth labels and predicted labels
Parameters: 'index' is index of image pair number, 'y_pred' model prediction, 'dataset' is dataset, 'gtruth_labels' ground truth labels'''
def draw_image_pairs(index,y_prediction,dataset,gtruth_labels): 

  fig = plt.figure()
  ax1 = fig.add_subplot(2,2,1)
  pred1 = 'same' if y_prediction[index] < 0.5 else 'differ'
  ground1 = 'same' if gtruth_labels[index] == 1 else 'differ'

  print('    True Label: ', ground1,' Predicted Label: ',pred1)
  
  ax1.imshow(dataset[index,0].reshape((28,28)),cmap='gray')
  ax1.set_yticklabels([])
  ax1.set_xticklabels([])
  ax2 = fig.add_subplot(2,2,2)
  ax2.imshow(dataset[index,1].reshape((28,28)),cmap='gray')
  ax2.set_yticklabels([])
  ax2.set_xticklabels([])
  plt.show()

#=================================#
#       End Functions Code here   #
#=================================#

#=================================#
#   Starts main Code from here    #
#=================================#


#ignore warnings
import warnings
warnings.filterwarnings("ignore")

#load data 
data,labels = load_fashionmnist_data()

#declaring list of labels that needs to be seperate in train and test data
train_set_labels = [0,1,2,4,5,9] # "top", "trouser", "pullover", "coat", "sandal", "ankle boot"
testlabels_list = [3,6,7,8] #"dress", "sneaker", "bag", "shirt"

#split data on basis of labels and make pairs 
#return: pair of images and labels=> 4 sets of data
#1 training set , 2 test1 set corresponds to 20% of training data , 3 test2 dataset corresponds to  ("dress", "sneaker", "bag", "shirt")
#4 test1utest2 union of test1 and test2 dataset
x1,y1,x2,y2,x3,y3,x4,y4 = split_and_make_pairs(data,labels,train_set_labels,testlabels_list)
train_x,train_y = (x1,y1) # pairs of images of train data of labels ( "top", "trouser", "pullover", "coat", "sandal", "ankle boot)"
test1_x,test1_y = (x2,y2) # pairs of images of 20% of train data of labels ( "top", "trouser", "pullover", "coat", "sandal", "ankle boot)"
test2_x,test2_y = (x3,y3) # All pairs data related to labels ("dress", "sneaker", "bag", "shirt")
test1utest2_x,test1utest2_y = (x4,y4) #Pair of images related to labels (["dress", "sneaker", "bag", "shirt"] union ["top", "trouser", "pullover", "coat", "sandal", "ankle boot])

#building siamese model with learningrate 0.00001 and rms prop optimizer
siamese_model = Build_siamese_model(train_x,learningRate=0.001)

#training model on train_x data and validation on test1_x data
batchsize = 256
epochs= 5
siamese_model,loss_acc_history = train(siamese_model,train_x,train_y,test1_x,test1_y,batchsize,epochs)

#plotting loss and accuracy curves with time
plt_loss_curve(loss_acc_history.history['loss'],loss_acc_history.history['val_loss'])
plt_acc_curve(loss_acc_history.history['model_accuracy'],loss_acc_history.history['val_model_accuracy'])

'''Testing on Different testing sets to calculate accuracy and generalization capacity'''
import matplotlib.pyplot as plt


# Testing on 'test1_x' dataset of labels ["top", "trouser", "pullover", "coat", "sandal", "ankle boot]
test1_score = print_accuracy(siamese_model,test1_x,test1_y,'Accuracy on test set 1')


# Testing on 'test2_x' dataset of labels ["dress", "sneaker", "bag", "shirt"]
test2_score = print_accuracy(siamese_model,test2_x,test2_y,'Accuracy on test set 2')

# Testing on 'test1utest2_x' dataset of labels ["top", "trouser", "pullover", "coat", "sandal", "ankle boot] union ["dress", "sneaker", "bag", "shirt"]
test1utest2_score = print_accuracy(siamese_model , test1utest2_x,test1utest2_y,'Accuracy on union of test1 and test2 dataset')

#Draw pair of images with ground truth and predicted label of test set 1.Change pair_number and see different pairs in every run
pair_number = 3
draw_image_pairs(pair_number,test1_score,test1_x,test1_y)

#Draw pair of images with ground truth and predicted label of test set 2.Change pair_number and see different pairs in every run
pair_number = 3
draw_image_pairs(pair_number,test2_score,test2_x,test2_y)

#Draw pair of images with ground truth and predicted label of union of test1 and test2 dataset.Change pair_number and see different pairs in every run
pair_number = 3
draw_image_pairs(pair_number,test1utest2_score,test1utest2_x,test1utest2_y)

