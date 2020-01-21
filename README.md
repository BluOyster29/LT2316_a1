# LT2316_a1

## Language Identification using RNN by incrementally adding characters
The repo contains the code for the above assignment given by Asad Sayeed of the University of Gothenburg machine learning course in the fall of 2019
   
## Goal
To correctly identify a language using the fewest characters possible 

## Main Scripts  

1. `pre_process.py`: This script allows the user to specify the languages or use a preset as well as choosing the batch size and naming the model
   * Arguments:
   * `-P`(y/n): Allows the user to choose preset or not (str)
   * `-B`: Specify batch size (int)
   * `-M`: Write a string to name the model (str)
2. `train.py`: Script for training model 
   * Arguments
   * `-E`: Enter epochs for training (int)
   * `-L`: Loss mode (int)
      * `1`: Character length loss not applied
      * `2`: Character length added to cross-entropy loss
      * `3`: Character length multiplied with cross-entropy loss
   * `-G` (y/n): Use GPU (str)
   * `-M`: Enter model name for training (same as from pre_process.py)
   
3. `test.py`: Script for testing model and displaying stats
   * `-M`: Enter model name for testing
   
### Helper Scripts 

1. `config.py`: Creates a json file for keeping track of model meta data
2. `GRUNetwork.py`: Pytorch class for generating RNN network
3. `LangIdentDataset.py`: Pytorch inbuilt class for creating dataset to be fed to dataloader function
4. `reset_outputs.py`: Script deletes all data created by the main functions. Acts as a clean reset. 
5. `stats.py`: Function for processing statics
## Part 1: Data Preparation 

### Choosing Languages 
Languages can either be loaded by defaul or the user can input their own, however they have to write them excactly how they are in the labels.csv (it is slightly buggy)
The default languages are: 
- English (eng)
- Belarusian (Taraschkewiza, Be-Tarask) (Belarusian)
- Karachay-Balkar (krc)
- Maithili (mai)
- Bokmål (nob)
- Western Panjabi (pnb)
- Sardinian (srd)
- Tetum (tet)
- Thai (tha)
- Xhosa (xho)
The default languages are a mixture of germanic, slavic and asian, I wanted to have some similar languages but also languages from different alphabets. 

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

Below are tables from free testing instances
1. Test 1: First test uses batching of 200, loss mode 1 trained over 10 epochs
2. Test 2:
3. Test 3:

| Language  | Avg Number of chars | Accuracy  | Num Char |
| --------- |-------------------- | --------- | ---------|
| eng      | right-aligned | $1600 |
| krc      | centered      |   $12 |
| mai | are neat      |    $1 |
| eng      | right-aligned | $1600 |
| krc      | centered      |   $12 |
| mai | are neat      |    $1 |

English (eng)
Belarusian (Taraschkewiza, Be-Tarask) (Belarusian)
Marachay-Balkar (krc)
- Maithili (mai)
- Bokmål (nob)
- Western Panjabi (pnb)
- Sardinian (srd)
- Tetum (tet)
- Thai (tha)
- Xhosa (xho)
