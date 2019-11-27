# LT2316_a1

## Language Identification using RNN by incrementally adding characters
The repo contains the code for the above assignment given by Asad Sayeed of the University of Gothenburg machine learning course in the fall of 2019

## Goal
To correctly identify a language using the fewest characters possible 

## Part 1: Data Preparation 

### Choosing Languages 
Languages can either be loaded by defaul or the user can input their own, however they have to write them excactly how they are in the labels.csv (it is slightly buggy)
The default languages are: 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
The default languages are a mixture of germanic, slavic and asian, I wanted to have some similar languages but also languages from different alphabets. 
The training/testing outputs are outputting after pre-processing. 

### Pre-Processing 
Once the languages have been split the program loads all the examples in the defined languages. These are then encoded using each character's index in the vocabulary.
Each sentence is the turned into it's numerical representation and outputted to '/data/pre_processed/'. 
The data is also loaded into pytorch data sets to be used in training. Once the data is turned into a DataLoader (for training set) and DataSet (for testing) they are pickled and saved in the folder 'dataloaders/'. The batch size is also defined at this stage, when the DataLoaders are being created one can specify the size of the batches and shuffle the data. All configurations for the network are saved in config.json that is located in the config/ folder
 
## Part 2: Training a Model

### Model
The neural network used is a Recurrant Nueral Network with Gated Recurrant Unit. The architecture is made using Pytorch and consists of an embedding layer, hidden layaer, gru and linear layer. All the adjustable paramaters are saved in config.json and the rest are hard coded. 

### Training Loop
If using batches then the training data is fed in batch by batch which does improve training time. If batch_size is not defined it is defaulted to 1 which takes much longer time. 
Once the model is trained it is pickled and saved in 'trained_models/' to be used in the evaluation stage  

## Part 3: Evaluation 

### Test Data
The test data has previously been split and turned into a DataLoader of 100 batches and not shuffled. This is so we can iterate through each example and break if the correct answer is found. 

### Evaluating 
The Model iterates through the test data and makes predictions, if the prediction is the correct label then the loop breaks and goes to the next example. Statistics are then calculated to find out accuracy, which languages are guessed, 

## Results 

