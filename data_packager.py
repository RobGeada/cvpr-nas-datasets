import collections
import numpy as np
import os
import json
import shutil
from gen_language_data import load_language_data
from gen_multnist_data import load_multnist_data, load_addnist_data
from gen_gutenberg import load_gutenberg
from gen_cifartile import load_cifartile_data

from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split as tts


# convert a list of tensors into a list of np arrays
def tlist_to_numpy(tlist):
    return [x.numpy() for x in tlist]


def process_torch_dataset(name, idx, location, verbose=False, return_data=False):
    # load various datasets, put into respective dirs
    metainfo = []
    if name == 'MultNIST':
        (train_x, train_y, _), (test_x, test_y) = load_multnist_data()
        metadata = {'batch_size': 64, 'n_classes': 10, 'lr': .01, 'benchmark': 91.55}
    elif name == 'AddNIST':
        (train_x, train_y, _), (test_x, test_y) = load_addnist_data()
        metadata = {'batch_size': 64, 'n_classes': 20, 'lr': .01, 'benchmark': 92.08}
    elif name == 'Language':
        (train_x, train_y), (test_x, test_y) = load_language_data(metainfo=False, verbose=False)
        metadata = {'batch_size': 64, 'n_classes': 10, 'lr': .01, 'benchmark': 87.00}
    elif name == 'Gutenberg':
        (train_x, train_y), (test_x, test_y) = load_gutenberg()
        print(train_x.shape, test_x.shape)
        metadata = {'batch_size': 64, 'n_classes': 6, 'lr': .01, 'benchmark': 40.98}
    elif name == 'CIFARTile':
        (train_x, train_y), (test_x, test_y) = load_cifartile_data()
        metadata = {'batch_size': 64, 'n_classes': 4, 'lr': .1, 'benchmark': 45.56}
    elif name == 'FashionMNIST':
        download = name not in os.listdir('raw_data')
        batch_size = 64
        train_data = datasets.FashionMNIST('raw_data/' + name,
                                           train=True,
                                           download=download,
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5,), (0.5,))]
                                           ))
        test_data = datasets.FashionMNIST('raw_data/' + name,
                                          train=False,
                                          download=download,
                                          transform=transforms.Compose([
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5,), (0.5,))]
                                          ))
        train_x, train_y = zip(*train_data)
        test_x, test_y = zip(*test_data)
        train_x = tlist_to_numpy(train_x)
        test_x = tlist_to_numpy(test_x)
        train_x, train_y = np.stack(train_x), np.array(train_y)
        test_x, test_y = np.stack(test_x), np.array(test_y)
        metadata = {'batch_size': 64, 'n_classes': 10, 'lr': .01, 'benchmark': 92.87}
    else:
        raise ValueError("Invalid dataset name!")

    # split train data into train and valid
    train_x, valid_x, train_y, valid_y = tts(train_x, train_y, train_size=45000, test_size=15000)

    # print out stats of label distribution across classes
    def sort_counter(c):
        return sorted(list(c.items()), key=lambda x: x[0]) 

    if verbose:
        print("=== {} ===".format(name))
        print('Train Bal:', sort_counter(collections.Counter(train_y)))
        print('Valid Bal:', sort_counter(collections.Counter(valid_y)))
        print('Test Bal:', sort_counter(collections.Counter(test_y)))

    # randomly shuffle arrays
    train_shuff = np.arange(len(train_y))
    valid_shuff = np.arange(len(valid_y))
    test_shuff = np.arange(len(test_y))
    np.random.shuffle(train_shuff)
    np.random.shuffle(valid_shuff)
    np.random.shuffle(test_shuff)
    train_x, train_y = train_x[train_shuff], train_y[train_shuff]
    valid_x, valid_y = valid_x[valid_shuff], valid_y[valid_shuff]
    test_x, test_y = test_x[test_shuff], test_y[test_shuff]

    # print out data shapes of each split
    if verbose:
       print("{} |  Train: {}, {} | Valid: {}, {} | Test: {}, {} |".format(
            name,
            train_x.shape, train_y.shape,
            valid_x.shape, valid_y.shape,
            test_x.shape, test_y.shape))

    # name and tag datasets
    dataset_type = location
    dataset_name = "{}_dataset_{}".format(dataset_type, idx)
    metadata['name'] = dataset_name

    # establish directory paths
    if location == 'eval':
        dataset_paths = ['evaluation/data/'+dataset_name]
    elif location == 'devel':
        dataset_paths = ['devel/data/'+dataset_name]

    if return_data:
        return [train_x, train_y], [valid_x,valid_y], [test_x, test_y], metadata

    for dataset_path in dataset_paths:
        if os.path.isdir(dataset_path):
            shutil.rmtree(dataset_path)

        os.mkdir(dataset_path)
        np.save(dataset_path+'/train_x.npy', train_x, allow_pickle=False)
        np.save(dataset_path+'/train_y.npy', train_y, allow_pickle=False)
        np.save(dataset_path+'/valid_x.npy', valid_x, allow_pickle=False)
        np.save(dataset_path+'/valid_y.npy', valid_y, allow_pickle=False)
        np.save(dataset_path+'/test_x.npy', test_x, allow_pickle=False)
        np.save(dataset_path+'/test_y.npy', test_y, allow_pickle=False)
        with open(dataset_path+'/metadata', "w") as f:
            json.dump(metadata, f)
    print("Processed {} dataset {}".format(dataset_type, idx))


if __name__ == '__main__':
    # load and save development datasets
    process_torch_dataset("AddNIST", 0, location='devel')       # development 0
    process_torch_dataset('FashionMNIST', 1, location='devel')  # development 1
    process_torch_dataset('Language', 2, location='devel')      # development 2

    process_torch_dataset("MultNIST", 0, location='eval')       # evaluation 0
    process_torch_dataset('CIFARTile', 1, location='eval')      # evaluation 1
    process_torch_dataset('Gutenberg', 2, location='eval')      # evaluation 2



