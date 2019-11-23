import pandas as pd
import os, pickle, get_data, argparse, gzip, torch
from os import listdir
from torch.nn.utils.rnn import pad_sequence
from LangIdentDataset import RTDataset
from torch.utils.data import DataLoader
from config import CONFIG, update_config

def get_args():
    parser = argparse.ArgumentParser(
        description="Pre Processing")
    parser.add_argument("-F", "--Folder", dest='folder', type=str, default="data/pre_processed/",
                        help="Folder containing outputted csv files")
    parser.add_argument("-B", "--batch_size", dest='batch_size', type=int, default=1,
                        help="Define the batch size for training")
    args = parser.parse_args()
    return args
import pickle
import get_data
import gzip

def load_csv(file_path):

    '''
    Loads all csvs from a file_path
    '''

    training = pd.read_csv(file_path + listdir(file_path)[0])
    testing = pd.read_csv(file_path + listdir(file_path)[1])
    x_train = [i for i in training['Language Example']]
    y_train = [i for i in training['Language index']]
    x_test = [i for i in testing['Language Example']]
    y_test = [i for i in testing['Language index']]
    language_codes = [i for i in list(set(training['Language index']))]

    return x_train, y_train, x_test, y_test, language_codes

def langencoder(language_codes):

    '''
    returns index for each language
    '''

    lang2int = {lang : (num) for num, lang in dict(enumerate(language_codes)).items()}

    return lang2int

def build_vocab(x_train):

    '''
    builds character to index in voab dictionary and also adds token for not in
    vocab tokens
    '''

    total_data = ''.join(x_train)
    int2char = dict(enumerate(set(total_data)))
    char2int = {char : num for num, char in int2char.items() }
    char2int['<niv>'] = max(char2int.values()) +1

    return char2int

def build_data(x,y, lang2int,vocab):

    '''
    Loops through text in dataset and encodes character to ingegral representation
    from the vocab and returns a padded tensor
    '''

    labels = []
    vectors = []
    sets = zip(x,y)
    for samples in sets:

        sample = [i for i in samples[0]]
        label = samples[1]
        #print(label)
        count = 100
        while count != 0:
            #x = [(i,int2label[samples[1]]) for i in samples[0]]
            vector = []
            encoded = []
            for i in sample:
                if i in vocab:
                    encoded.append(vocab[i])
                else:
                    encoded.append(vocab['<niv>'])
            for i in range(1,101):
                vectors.append(torch.LongTensor(encoded[:i])) #, int(lang2int[label])))
                labels.append(lang2int[label])
                count -=1

    return pad_sequence(vectors, batch_first=True, padding_value=0), labels

def output_postprocessed(train_dataset,test_dataset):

    '''
    Not really necessary, pd dataframes don't preserve datatype
    '''

    dir = 'data/postprocessed/'
    if os.path.exists(dir) == False:
        os.mkdir(dir)
    directory = 'data/postprocessed/'
    if os.path.exists(directory) == False:
        os.mkdir(directory)

    pd.DataFrame.to_csv(pd.DataFrame({'Encoded Language Example' : [i[0] for i in train_dataset],
                           'Language Index' : [i[1] for i in train_dataset]}), 'data/postprocessed/post_processed_training.csv')

    pd.DataFrame.to_csv(pd.DataFrame({'Encoded Language Example' : [i[0] for i in test_dataset],
                           'Language Index' : [i[1] for i in test_dataset]}),'data/postprocessed/post_processed_testing.csv')

def output_dataloaders(loaders):

    '''''
    Outputting Dataloaders to be used in training and testing
    '''

    directory = 'dataloaders/'
    if os.path.exists(directory) == False:
        os.mkdir(directory)

    training = 'dataloaders/training_dataloader.zip'
    testing = 'dataloaders/testing_dataset.zip'
    for i in zip([training,testing], loaders):
    training = 'dataloaders/training_dataloader.pkl'
    testing = 'dataloaders/testing_dataloader.pkl'


    for i in zip([training, testing], loaders):
        print('Pickling Dataloader {}'.format(str(i[0])))
        file = gzip.GzipFile(i[0], 'wb')
        file.write(pickle.dumps(i[1], 1))
        file.close()
        with open(i[0], 'wb') as file:
            pickle.dump(i[1], file)

    return 'Done'''

def save_dataloaders(train_loader, test_loader):
    directory = 'dataloaders/'
    if os.path.exists(directory) == False:
        os.mkdir(directoy)

    train = 'training_loader.pkl'
    test = 'test_dataset.pkl'

    for i in zip([train_loader, test_loader], [train,test]):
        with open(directory+i[1], 'wb') as file:
            pickle.dump(i[0],file)

def pre_process_main():
    args = get_args()
    '''
    x_train, y_train, vocab, int2char, x_test, y_test = get_data.get_data_main()
    dir = 'data/pre_processed/'
    if os.path.exists(dir) == False:
        os.mkdir(dir)'''
    print('Loading Csvs')
    x_train, y_train, x_test, y_test, language_codes = load_csv(args.folder)
    directory = 'data/pre_processed/'
    x_train, y_train, x_test, y_test, language_codes = load_csv(directory)
    print('Building Vocab')
    vocab = build_vocab(x_train)
    output_vocab(vocab)
    lang2int = langencoder(language_codes)
    print('Preprocessing training data')
    train_data, train_labels = build_data(x_train, y_train, lang2int, vocab)
    print('Preprocessing testing data')
    test_data, test_labels = build_data(x_test, y_test, lang2int, vocab)
    train_dataset = RTDataset(train_data, train_labels)
    test_dataset = RTDataset(test_data, test_labels)
    CONFIG['batch_size'] = args.batch_size
    update_config(CONFIG)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    print('Outputting dataloaders')
    save_dataloaders(train_loader,test_dataset)
    #output_postprocessed(train_dataset, test_dataset)
    #output_dataloaders([train_loader,test_dataset])
    print('Done!')

def output_vocab(vocab):
    directory = 'vocab/'
    if os.path.exists(directory) == False:
        os.mkdir(directory)
    print('Outputting vocab to {}'.format(directory+'vocab.pkl'))
    with open('{}vocab.pkl'.format(directory), 'wb') as file:
        pickle.dump(vocab, file)
        file.close()

if __name__ == '__main__':
    pre_process_main()
