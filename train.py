import gzip, shutil, torch, pickle, os
from GRUNetwork import RNN_GRU
from config import CONFIG, update_config
import argparse
from tqdm import tqdm_notebook as tqdm
from torch import nn
from torch.optim import Adam
import torch.optim as optim
import sys


def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

def get_args():
    parser = argparse.ArgumentParser(
        description="Training")
    parser.add_argument("-F", "--Folder", dest='folder', type=str, default="data/pre_processed/",
                        help="Folder containing outputted csv files")
    parser.add_argument("-B", "--batch_size", dest='batch_size', type=int, default=1,
                        help="Define the batch size for training")
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
		print(filename)
		with open(filepath+filename, 'rb') as file:
			dataloaders.append(pickle.load(file))
	return dataloaders[0], dataloaders[1]

def save_model(model, model_nr):
    directory = 'trained_models/'
    if os.path.exists(directory) == False:
        os.mkdir(directory)
    torch.save(model.state_dict(), '{}gru_model_nr{}.pt'.format(directory,model_nr))

def get_vocab(path):
    with open(path+'vocab.pkl', 'rb')as file:
        vocab = pickle.load(file)
    return vocab

def train(model, train_loader, vocab_size, device, nr_of_epochs, batch_size):


	criterion = nn.CrossEntropyLoss()
	optimizer = Adam(model.parameters(), lr=0.001)
	model.train()
	model = model.to(device)
	print('Training')
	epoch_nr = 0
	EPOCH = list(range(nr_of_epochs))

	for epoch in EPOCH:

		epoch_nr += 1
		epoch_loss = []
		h = model.init_hidden(batch_size)
		count = 0
		total = 50000000
		percent = 10
		progress(count, total, status='training')

		for (x,y) in train_loader:
			progress(count, total, status='trainings')
			print(count)
			count +=1
			x = x.to(device)
			y = y.to(device)
			optimizer.zero_grad()
			h = h.data
			out, h = model(x, h)
			loss = criterion(out, y.long())
			loss.backward()
			epoch_loss.append(loss.item())
			optimizer.step()
			#print('Loss per timestep = {}'.format(loss.item()))
			progress(count, total, status='training')
		avg_loss = sum(epoch_loss) / len(epoch_loss)
		print("Average loss at epoch %d: %.7f" % (epoch_nr, avg_loss))

if __name__ == '__main__':
	device = torch.device(CONFIG['device'] if torch.cuda.is_available() else "cpu")
	vocab_size = len(get_vocab('vocab/'))
	#unpack_dataloaders('dataloaders/')
	training, testing = load_dataloaders('dataloaders/')
	batch_size = CONFIG['batch_size']
	model = RNN_GRU(vocab_size=vocab_size, seq_len=CONFIG['sequence_length'],
	               input_size=CONFIG['sequence_length'], hidden_size=CONFIG['hidden_size'],
				   num_layers=CONFIG['number_layers'], output_size=CONFIG['output_size'],
				   device=device, dropout=0.0)
	nr_of_epochs = 5
	trained_model = train(model, training, vocab_size, device, nr_of_epochs, batch_size)
