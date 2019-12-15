import gzip, shutil, torch, pickle, os, config, argparse
from GRUNetwork import RNN_GRU
from tqdm import tqdm_notebook as tqdm
from torch import nn
from torch.optim import Adam
import torch.optim as optim
import sys,json

def get_args():
    parser = argparse.ArgumentParser(
        description="Training")

    parser.add_argument("-E", "--nr_epochs", dest='nr_epochs', type=int,
                        default=2, help="Define the number of training epochs")
    parser.add_argument("-G", "--use_gpu", dest='use_gpu', type=str,
                        help='Use GPU or CPU for training (y/n)', default="y")
    args = parser.parse_args()
    return args

def unpack_dataloaders(filepath):
    dataloaders = []

    for filename in os.listdir(filepath):
        with gzip.open(filepath+filename, 'rb') as file_in:
            with open(filepath+filename[:-4]+'.pkl', 'wb') as file_out:
                shutil.copyfileobj(file_in, file_out)
                dataloaders.append(file_out)
        os.remove(filepath+filename)

def load_dataloaders(filepath):
    loaders = os.listdir(filepath)
    dataloaders = []

    for filename in os.listdir(filepath):
        with open(filepath+filename, 'rb') as file:
            dataloaders.append(pickle.load(file))
    return dataloaders[0], dataloaders[1]

def save_model(model, batch_size, nr_epochs):
    directory = 'trained_models/'
    if os.path.exists(directory) == False:
        os.mkdir(directory)
    torch.save(model.state_dict(), '{}gru_model_{}batches_{}epochs.pt'.format(directory,batch_size,nr_epochs))

def get_vocab(path):
    with open(path+'vocab.pkl', 'rb')as file:
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
    tenp = len(train_loader) / 10
    for epoch in EPOCH:

        epoch_nr += 1
        epoch_loss = []
        h = model.init_hidden(batch_size)
        count = 0
        percent = 0

        for (x,y) in train_loader:
            count +=1
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            h = h.data
            out, h = model(x, h)
            if loss_mode == 1:
                loss = criterion(out, y.long())
            elif loss_mode == 2:
                continue
            elif loss_mode == 3:
                continue
            loss.backward()
            epoch_loss.append(loss.item())
            optimizer.step()
            if count % tenp == 0:
                percent += 10
                print("Training {}% complete".format(round(percent,1)))

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
    vocab_size = len(get_vocab('vocab/'))
    #unpack_dataloaders('dataloaders/')
    training, testing = load_dataloaders('dataloaders/')
    batch_size = CONFIG['batch_size']
    model = RNN_GRU(vocab_size=vocab_size, seq_len=CONFIG['sequence_length'],
                   input_size=CONFIG['sequence_length'], hidden_size=CONFIG['hidden_size'],
                   num_layers=CONFIG['number_layers'], output_size=CONFIG['output_size'],
                   device=torch.device(device), dropout=0.0)
    print('Model Generated')
    nr_of_epochs = args.nr_epochs
    print('Training Model with {} batches over {} epochs using loss mode {} on {}'.format(batch_size,nr_of_epochs, device))
    trained_model = train(model, training, vocab_size, device, nr_of_epochs, batch_size, loss_mode)
    print('Model Trained')
    save_model(model,batch_size,nr_of_epochs)

if __name__ == '__main__':
    args = get_args()
    main(args)
