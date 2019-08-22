import os

def make_dirs(root, path):
    
    directories = [
        ['checkpoints'],
        ['predictions'],
        ['figures', 'snapshots'],
    ]

    for paths in directories:
        full_dir = path
        for folder in paths:
            full_dir = os.path.join(full_dir, folder)

        if not os.path.isdir(full_dir):
            print(f'{full_dir} does not exist. Creating path...')
            os.makedirs(full_dir)
        else:
            print(f'{full_dir} already exists')
