library(reticulate)
#install_miniconda(force = TRUE)
library(keras)
library(dplyr)
library(EBImage)
library(BiocManager)

#install_keras()

#BiocManager::install("EBImage") 

#set working directory 

#setwd("D:/STUDENTS/Haritha/Pratheesh/raw")
#card<-readImage("june_3.jpg")
#print(card)
#getFrames(card, type = "total")
#display(card)

### access all ripe
setwd("D:/MSC STAT STUDENTS/2021 batch/ARUNIMA/CNN_banana/Pratheesh/ripe")
img.card<- sample(dir()); #-------shuffle the order
cards<-list(NULL);        
for(i in 1:length(img.card))
{ cards[[i]]<- readImage(img.card[i])
cards[[i]]<- resize(cards[[i]], 100, 100)} #resizing to 100x100
ripe<- cards      # Storing stack of images in # matrix form in a list
#-----------------------------------------------------------

### access all raw
setwd("D:/MSC STAT STUDENTS/2021 batch/ARUNIMA/CNN_banana/Pratheesh/raw")
img.card<- sample(dir()); #-------shuffle the order
cards<-list(NULL);        
for(i in 1:length(img.card))
{ cards[[i]]<- readImage(img.card[i])
cards[[i]]<- resize(cards[[i]], 100, 100)} #resizing to 100x100
raw<- cards      # Storing stack of images in # matrix form in a list
#-----------------------------------------------------------

### access all medium
setwd("D:/MSC STAT STUDENTS/2021 batch/ARUNIMA/CNN_banana/Pratheesh/medium")
img.card<- sample(dir()); #-------shuffle the order
cards<-list(NULL);        
for(i in 1:length(img.card))
{ cards[[i]]<- readImage(img.card[i])
cards[[i]]<- resize(cards[[i]], 100, 100)} #resizing to 100x100
medium<- cards      # Storing stack of images in # matrix form in a list
#-----------------------------------------------------------

### access all above medium
setwd("D:/MSC STAT STUDENTS/2021 batch/ARUNIMA/CNN_banana/Pratheesh/abv_med")
img.card<- sample(dir()); #-------shuffle the order
cards<-list(NULL);        
for(i in 1:length(img.card))
{ cards[[i]]<- readImage(img.card[i])
cards[[i]]<- resize(cards[[i]], 100, 100)} #resizing to 100x100
abv_medium<- cards      # Storing stack of images in # matrix form in a list
#-----------------------------------------------------------

###Training Set
train_pool<-c(ripe[1:3], 
              raw[1:3], 
              medium[1:3], 
              abv_medium[1:3]) # The first 3 images from each
                                # are included in the train set

train<-aperm(combine(train_pool), c(4,1,2,3)) # Combine and stacked


#### Test set
test_pool<-c(ripe[4:5], 
             raw[4:5], 
             medium[4:5], 
             abv_medium[4:5])

test<-aperm(combine(test_pool), c(4,1,2,3)) # Combined and stacked

##See images in test set
par(mfrow=c(3,4)) # To contain all images in single frame
for(i in 1:8){
  plot(test_pool[[i]])
}
par(mfrow=c(1,1)) # Reset the default

###### hot encoding for categorical data
#one hot encoding
train_y<-c(rep(0,3),rep(1,3),rep(2,3),rep(3,3))
test_y<-c(rep(0,2),rep(1,2),rep(2,2),rep(3,2))



#####
train_lab<-to_categorical(train_y) #Catagorical vector for training 
#classes
test_lab<-to_categorical(test_y)#Catagorical vector for test classes


# Model Building
model.card<- keras_model_sequential() #-Keras Model composed of a 
#-----linear stack of layers
model.card %>%  
  #---------Initiate and connect to #----------------------------(A)-----------------------------------#
  
  layer_conv_2d(filters = 100,       #----------First convoluted layer
                kernel_size = c(4,4),             #---40 Filters with dimension 4x4
                activation='relu',           #-with a ReLu activation function
                input_shape = c(100,100,3)) %>%   
  #----------------------------(B)-----------------------------------#
  layer_conv_2d(filters = 80,       #---------Second convoluted layer
                kernel_size = c(4,4),             #---40 Filters with dimension 4x4
                activation='relu') %>% 
  
  #---------------------------(C)-----------------------------------#
  layer_max_pooling_2d(pool_size = c(4,4) )%>%   #--------Max Pooling
  #-----------------------------------------------------------------#
  layer_dropout(rate = 0.25) %>%   #-------------------Drop out layer
  #----------------------------(D)-----------------------------------#
  layer_conv_2d(filters = 80,      #-----------Third convoluted layer
                kernel_size = c(4,4),            #----80 Filters with dimension 4x4
                activation='relu')%>%        #--with a ReLu activation function 
  
        #--with a ReLu activation function
  #-----------------------------(E)----------------------------------#
  layer_conv_2d(filters = 80,      #----------Fourth convoluted layer
                kernel_size = c(4,4),            #----80 Filters with dimension 4x4
                activation='relu') %>%         #--with a ReLu activation function
  
  #-----------------------------(F)----------------------------------#
  layer_max_pooling_2d(pool_size = c(4,4)) %>%  #---------Max Pooling
  
  #-----------------------------------------------------------------#
  layer_dropout(rate = 0.35) %>%   #-------------------Drop out layer
  
  #------------------------------(G)---------------------------------#
  layer_flatten()%>%   #---Flattening the final stack of feature maps
  
  #------------------------------(H)---------------------------------#
  layer_dense(units = 500, activation='relu')%>% #-----Hidden layer
  #---------------------------(I)-----------------------------------#
  layer_dropout(rate= 0.25)%>%     #-------------------Drop-out layer
  #-----------------------------------------------------------------#
  layer_dense(units = 4, activation="softmax")%>% #-----Final Layer
 #----------------------------(J)-----------------------------------#
  compile(loss = 'categorical_crossentropy',
          optimizer = optimizer_adam(),
          metrics = c("accuracy"))   # Compiling the architecture


model.card %>% summary()



#fit model
history<- model.card %>%
  fit(train, 
      train_lab, 
      epochs = 100,
      batch_size = 4,
      validation_split = 0.2
      
  )

model.card %>% evaluate(train,train_lab) #Evaluation of training set pred<- model.card %>% predict_classes(train) #-----Classification
pred<-model.card %>% predict(train) %>% k_argmax()
pred<-as.matrix(pred)
Train_Result<-table(Predicted = pred, Actual = train_y)

model.card %>% evaluate(test, test_lab) #-----Evaluation of test set
pred1<-model.card %>% predict(test) %>% k_argmax()
pred1<- pred<-as.matrix(pred1)
Test_Result<-table(Predicted = pred1, Actual = test_y) #-----Results


