import gzip, shutil, torch, pickle, os, config, argparse
from GRUNetwork import RNN_GRU
from tqdm import tqdm
from torch import nn
from torch.optim import Adam
import torch.optim as optim
import sys,json

def get_args():
    parser = argparse.ArgumentParser(
        description="")
    parser.add_argument("-D", "--Data", dest='data', type=str, default="data/raw/",
                        help="Directory that contains training data")
    parser.add_argument("-B", "--batch_size", dest='batch_size', type=int, default=1,
                        help="Define the batch size for training")
    parser.add_argument("-E", "--nr_epochs", dest='nr_epochs', type=int,
                        default=2, help="Define the number of training epochs")
    parser.add_argument("-G", "--use_gpu", dest='use_gpu', type=str,
                        help='Use GPU or CPU for training (y/n)', default="y")
    parser.add_argument("-L", "--loss_mode", dest='loss_mode', type=int,
                        help='Choose loss mode', default="1")
    parser.add_argument("-M", "--model_name", dest='model_name', type=str,
                        help='Create name for model')

    args = parser.parse_args()
    return args

def unpack_dataloaders(filepath, model_name):
    dataloaders = []

    for filename in os.listdir(filepath):
        with gzip.open(filepath+filename, 'rb') as file_in:
            with open(filepath+filename[:-4]+'.pkl', 'wb') as file_out:
                shutil.copyfileobj(file_in, file_out)
                dataloaders.append(file_out)
        os.remove(filepath+filename)

def load_dataloaders(filepath):


    with open('dataloaders/{}{}'.format(filepath,'_training_loader.pkl'), 'rb') as file:
        dataloaders = pickle.load(file)

    return dataloaders

def save_model(model, batch_size, nr_epochs, model_name):
    directory = 'trained_models/'
    if os.path.exists(directory) == False:
        os.mkdir(directory)
    torch.save(model.state_dict(), 'trained_models/{}.pt'.format(model_name))

def get_vocab(path):
    with open(path, 'rb')as file:
        vocab = pickle.load(file)
    return vocab

def train(model, train_loader, vocab_size, device, nr_of_epochs, batch_size, loss_mode):
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    model.train()
    model = model.to(device)
    print('Training')
    epoch_nr = 0
    EPOCH = list(range(nr_of_epochs))
    tenp = round(len(train_loader) / 10)
    for epoch in tqdm(EPOCH):
        epoch_nr += 1
        epoch_loss = []
        h = model.init_hidden(batch_size)
        count = 0
        percent = 0

        for (x,y) in tqdm(train_loader):
            count +=1
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            h = h.data
            out, h = model(x, h)
            num_chars = len(torch.nonzero(out))
            if loss_mode == 1:
                loss = criterion(out, y.long())
            elif loss_mode == 2:
                loss = criterion(out, y.long()) + num_chars
    
            elif loss_mode == 3:
                loss = criterion(out, y.long()) * num_chars

            loss.backward()
            epoch_loss.append(loss.item())
            optimizer.step()

        avg_loss = sum(epoch_loss) / len(epoch_loss)
        print("Average loss at epoch %d: %.7f" % (epoch_nr, avg_loss))

    return model

def print_config(config):
    print('Training using the following configuartion')
    print(config)

def main(args):
    loss_mode = args.loss_mode
    CONFIG = config.get_config('config/config.json')
    if args.use_gpu == 'y':
        device = 'cuda:1'
    else:
        device = 'cpu'
    CONFIG['device'] = device
    config.update_config(CONFIG)
    vocab_size = len(get_vocab('vocab/{}{}'.format(args.model_name,'_vocab.pkl')))
    #unpack_dataloaders('dataloaders/')
    training = load_dataloaders(args.model_name)
    batch_size = CONFIG['batch_size']
    print('Creating model')
    model = RNN_GRU(vocab_size=vocab_size, seq_len=CONFIG['sequence_length'],
                   input_size=CONFIG['sequence_length'], hidden_size=CONFIG['hidden_size'],
                   num_layers=CONFIG['number_layers'], output_size=CONFIG['output_size'],
                   device=torch.device(device), dropout=0.0)
    print('Model Generated')
    nr_of_epochs = args.nr_epochs
    print('Training Model with {} batches over {} epochs using loss mode {} on {}'.format(batch_size,nr_of_epochs,str(loss_mode), device))
    trained_model = train(model, training, vocab_size, device, nr_of_epochs, batch_size, loss_mode)
    print('Outputting model to trained_model/{}.pt'.format(args.model_name))
    save_model(trained_model,batch_size,nr_of_epochs, args.model_name)
    return model

if __name__ == '__main__':
    args = get_args()
    main(args)
