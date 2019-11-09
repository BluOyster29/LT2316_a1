import os
import pickle

def load_dataloaders(filepath):
	"""Loads a compressed object from disk
	"""
    dataloaders = []
    for filename in listdir(filepath):
    	file = gzip.GzipFile(filename, 'rb')
    	buffer = ""
    	while 1:
    		data = file.read()
    		if data == "":
    			break
    		buffer += data
    	object = pickle.loads(buffer)
    	file.close()
	dataloaders.append(object)

    return dataloaders[0], dataloaders[1]

def get_dataloaders(filepath):
    dataloaders = []
    for i in os.listdir(filepath):
        with open(filepath + i, "rb") as input_file:
            dataloaders.append(pickle.load(input_file))
            input_file.close()
    return dataloaders[0], dataloaders[1]

def train(training_dataloader, model):
    pass

if __name__ ==
training, testing = load_dataloaders('dataloaders/')
