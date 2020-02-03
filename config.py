import torch, json, os

def gen_config(path):

    directory = 'config/'
    
    if os.path.exists(directory) == False:
        os.mkdir(directory)
    
    EMPTY_CONFIG = {'vocab_size'     : int(),
               'sequence_length' : 100,
               'batch_size'      : 1,
               'input_size'      : 100,
               'hidden_size'     : 256,
               'number_layers'   : 2,
               'output_size'     : 10,
               'device'          : str(),
               'languages'       : list()
                   }

    with open('config/config.json', 'w') as outfile:
        json.dump(EMPTY_CONFIG, outfile, sort_keys=True, indent=3)
    return EMPTY_CONFIG

def update_config(config_file):
    with open('config/config.json', 'w') as outfile:
        json.dump(config_file, outfile, sort_keys=True, indent=3)

def get_config(config_file):
    with open(config_file) as json_file:
        config = json.load(json_file)
    return config
