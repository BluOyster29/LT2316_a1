import os

def delete_everything(list_of_paths):
    for i in list_of_paths:
        files = os.listdir(i)
        for a in files:
            os.remove(i+a)
                    
delete_everything(['dataloaders/', 'trained_models/', 'vocab/', 'config/'])
