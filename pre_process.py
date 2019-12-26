import pandas as pd, csv
import os, pickle, get_data, argparse, gzip, torch, config
from os import listdir
from torch.nn.utils.rnn import pad_sequence
from LangIdentDataset import RTDataset
from torch.utils.data import DataLoader

def get_args():
    
    parser = argparse.ArgumentParser(
        description="")
    
    parser.add_argument("-P", "--preset", dest='preset', type=str,
                        help="Choose to use default language set or your own", default="y")
    parser.add_argument("-D", "--Data", dest='data', type=str, default="data/raw/",
                        help="Directory that contains training data")
    parser.add_argument("-B", "--batch_size", dest='batch_size', type=int, default=1,
                        help="Define the batch size for training")
    parser.add_argument("-F", "--Folder", dest='folder', type=str, default="data/pre_processed/",
                        help="Folder containing outputted csv files")
    parser.add_argument("-M", "--model_name", dest='model_name', type=str,
                        help='Create name for model')
    
    args = parser.parse_args()
    
    return args

def get_files_from_folder(folder):
    files = os.listdir(folder)
    
    return folder+files[-1], folder+files[1], folder+files[2], folder+files[3], folder+files[4]

def get_languages(csv_file, preset):
    with open(csv_file, 'r') as csv_File: #opens csv containing language codes and their names
        reader = csv.reader(csv_File)
        language_table = {row[0].split(';')[1] : row[0].split(';')[0] for row in reader} #dictionary mapping code to name
    if preset == 'y':
        #Preset language codes to be used in model, chosen at random
        language_codes = ["srd", "krc", "nob", "pnb",
                          "mai", "eng", "be-tarask",
                          "xho", "tet", "tha"]
        language_names = [(key, value) for key, value in language_table.items() if value in language_codes]
        return language_names
    
    elif preset == 'n':
        
        '''
        experimental function for allowing user to choose which languages to use
        '''
        
        languages = []
        while len(languages) != 10:
            language = input("Enter language ").capitalize()
            #print(language)
            if language in languages:
                print("You've already said that one! ")
            elif language in language_table:
                languages.append(language)
                print(' '.join(languages))
            else:
                print('Language not recognised. Please refer to language labels')
                print(' '.join(languages))
                continue
        language_codes = [language_table[i] for i in language_table if i in languages] #collects languages from predetermined set,
        language_names =  [(key, value) for key, value in language_table.items() if value in language_codes] #dictionary mapping code to name
        return language_names
    else:
        print("has to be y or n dummy") #in case of user error

def gen_data(training_file, training_labels, language_codes, training):
    '''
    Function generates the set based on pre defined language codes and creates various
    attributes to the object
    '''
    if training == True:
        data = [i.split('\n')[:-1] for i in open(training_file, 'r')] #opens text file and splits on new line
        labels = [i.split('\n')[:-1] for i in open(training_labels, 'r')] #opens label file and splits on white space
        things = list(zip([i[0] for i in data], [i[0] for i in labels])) #zips sentences with corrosponding language label
        sets = [(i[0],i[1]) for i in things] #this might actually do the same thing as the above not sure

        x = [i[0][:100] for i in sets if i[1] in language_codes] #Matrix of sentences to be used in the model
        y = [i[1] for i in sets if i[1] in language_codes] #labels for each of the sentences
        raw_data = ''.join([i for i in x]) #concatenation of all characters in the training set
        vocab = {char: ord(char) for char in set(raw_data)} #dictionary mapping character to ord(integer)
        int2char = {num : char for char, num in vocab.items()} #dictionary mapping integer to character
        return x, y, vocab, int2char
    else:
        data = [i.split('\n')[:-1] for i in open(training_file, 'r')] #opens text file and splits on new line
        labels = [i.split('\n')[:-1] for i in open(training_labels, 'r')] #opens label file and splits on white space
        things = list(zip([i[0] for i in data], [i[0] for i in labels])) #zips sentences with corrosponding language label
        sets = [(i[0],i[1]) for i in things] #this might actually do the same thing as the above not sure
        x = [i[0][:100] for i in sets if i[1] in language_codes] #Matrix of sentences to be used in the model
        y = [i[1] for i in sets if i[1] in language_codes] #labels for each of the sentences
        return x, y

def output_data(x, y, filename):
    directory = 'data/pre_processed/'

    if os.path.exists(directory) == False:
        os.mkdir(directory)

    output = pd.DataFrame(data={'Language Example' : x, 'Language index' : y})
    pd.DataFrame.to_csv(output, directory+filename)

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

def output_postprocessed(train_dataset,test_dataset,model_name):

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
                           'Language Index' : [i[1] for i in train_dataset]}), 'data/postprocessed/{}_post_processed_training.csv'.format(model_name))

    pd.DataFrame.to_csv(pd.DataFrame({'Encoded Language Example' : [i[0] for i in test_dataset],
                           'Language Index' : [i[1] for i in test_dataset]}),'data/postprocessed/{}_post_processed_testing.csv'.format(model_name))

def output_dataloaders(loaders, model_name):

    '''''
    Outputting Dataloaders to be used in training and testing
    '''

    directory = 'dataloaders/'
    if os.path.exists(directory) == False:
        os.mkdir(directory)

    training = 'dataloaders/{}_training_dataloader.pkl'.format(model_name)
    testing = 'dataloaders/{}_testing_dataloader.pkl'.format(model_name)
    for i in zip([training,testing], loaders):
        training = 'dataloaders/{}_training_dataloader.pkl'.format(model_name)
        testing = 'dataloaders/{}_testing_dataloader.pkl'.format(model_name)


    for i in zip([training, testing], loaders):
        print('Pickling Dataloader {}'.format(str(i[0])))
        file = gzip.GzipFile(i[0], 'wb')
        file.write(pickle.dumps(i[1], 1))
        file.close()
        with open(i[0], 'wb') as file:
            pickle.dump(i[1], file)

    return 'Done'''

def save_dataloaders(train_loader, test_loader, model_name):
    directory = 'dataloaders/'
    if os.path.exists(directory) == False:
        os.mkdir(directory)

    train = '{}_training_loader.pkl'.format(model_name)
    test = '{}_test_dataset.pkl'.format(model_name)

    for i in zip([train_loader, test_loader], [train,test]):
        with open(directory+i[1], 'wb') as file:
            pickle.dump(i[0],file)

def output_vocab(vocab, model):
    directory = 'vocab/'
    if os.path.exists(directory) == False:
        os.mkdir(directory)
    print('Outputting vocab to {}{}{}'.format(directory,model,'_vocab.pkl'))
    with open('{}{}_vocab.pkl'.format(directory, model), 'wb') as file:
        pickle.dump(vocab, file)
        file.close()

def main(args):
    CONFIG = config.gen_config('config/config.json')
    labels, x_test, x_train,y_test,y_train = get_files_from_folder(args.data)
    language_names = get_languages(labels, args.preset)
    language_codes = [i[1] for i in language_names]
    CONFIG['languages'] += language_names
    x_train, y_train, vocab, int2char = gen_data(x_train, y_train, language_codes, training=True)
    x_test, y_test = gen_data(x_test, y_test, language_codes, training=False)
    output_data(x_train,y_train, 'Training_data.csv')
    output_data(x_train,y_train, 'Testing_data.csv')
    config.update_config(CONFIG)
    directory = 'data/pre_processed/'
    if os.path.exists(directory) == False:
        os.mkdir(directory)
    print('Loading Csvs')
    x_train, y_train, x_test, y_test, language_codes = load_csv(args.folder)
    print('Building Vocab')
    vocab = build_vocab(x_train)
    output_vocab(vocab, args.model_name)
    lang2int = langencoder(language_codes)
    print('Preprocessing training data')
    train_data, train_labels = build_data(x_train, y_train, lang2int, vocab)
    print('Preprocessing testing data')
    test_data, test_labels = build_data(x_test, y_test, lang2int, vocab)
    train_dataset = RTDataset(train_data, train_labels)
    test_dataset = RTDataset(test_data, test_labels)
    CONFIG = config.get_config('config/config.json')
    CONFIG['batch_size'] = args.batch_size
    CONFIG['vocab_size'] = len(vocab)
    config.update_config(CONFIG)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    print('Outputting dataloaders')
    save_dataloaders(train_loader,test_dataloader, args.model_name)
    print('Done!')

if __name__ == '__main__':
    args = get_args()
    main(args)
