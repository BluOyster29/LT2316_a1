import torch, json, os

CONFIG = {'vocab_size'     : int(),
         'sequence_length' : 100,
         'batch_size'      : 1,
         'input_size'      : 100,
         'hidden_size'     : 256,
         'number_layers'   : 2,
         'output_size'     : 10,
         'device'          : 'cuda:01'
         }

def update_config(config_file):

    with open('config/config.json', 'w') as outfile:
        json.dump(CONFIG, outfile, sort_keys=True, indent=3)


