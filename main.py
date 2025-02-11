# everything in this main file is just using library code
# the additions to make the library work with deep learning are in the
# extra.py file.

from libemg.datasets import OneSubjectMyoDataset
from libemg.data_handler import OfflineDataHandler
from libemg.filtering import Filter
from libemg.emg_predictor import EMGClassifier
from libemg.offline_metrics import OfflineMetrics
from deeplearningspecificcode import fix_random_seed, make_data_loader, CNN

import numpy as np

def main():
    # make our results repeatable
    fix_random_seed(seed_value=0, use_cuda=True)
    # download the dataset from the internet
    dataset = OneSubjectMyoDataset()
    data = dataset.prepare_data()

    # split the dataset into a train, validation, and test set
    # this dataset has a "sets" metadata flag, so lets split 
    # train/test using that.
    not_test_data = data['Train']
    test_data = data['Test']
    # lets further split up training and validation based on reps
    train_data = not_test_data.isolate_data("sets",[0,1,2,3])
    valid_data = not_test_data.isolate_data("sets",[4])

    # let's perform the filtering on the dataset too (neural networks like
    # inputs that are standardized).
    fi = Filter(sampling_frequency=200)
    standardize_dictionary = {"name":"standardize",
                              "data": train_data}
    fi.install_filters(standardize_dictionary)
    fi.filter(train_data)
    fi.filter(valid_data)
    fi.filter(test_data)

    # for each of these dataset partitions, lets get our windows ready
    window_size, window_increment = 50, 10
    train_windows, train_metadata = train_data.parse_windows(window_size, window_increment)
    valid_windows, valid_metadata = valid_data.parse_windows(window_size, window_increment)
    test_windows,  test_metadata  = test_data.parse_windows( window_size, window_increment)

    #--------------------------------------#
    # Now we need to interface custom code #
    #--------------------------------------#
    # libemg supports deep learning, but we need to prepare the dataloaders
    train_dataloader = make_data_loader(train_windows, train_metadata["classes"])
    valid_dataloader = make_data_loader(valid_windows, valid_metadata["classes"])
    
    # let's make the dictionary of dataloaders
    dataloader_dictionary = {"training_dataloader": train_dataloader,
                             "validation_dataloader": valid_dataloader}
    # We need to tell the libEMG EMGClassifier that we are using a custom model
    model = CNN(n_output   = np.unique(np.vstack(not_test_data.classes[:])).shape[0],
                n_channels = train_windows.shape[1],
                n_samples  = train_windows.shape[2],
                n_filters  = 64)
    # we can even make a dictionary of parameters that get passed into the training
    # process of the deep learning model
    dl_dictionary = {"learning_rate": 1e-4,
                     "num_epochs": 50,
                     "verbose": True}
    model.fit(dataloader_dictionary, **dl_dictionary)
    #--------------------------------------#
    #          Back to library code        #
    #--------------------------------------#
    # Now that we've made the custom classifier object, libEMG knows how to 
    # interpret it when passed in the dataloader_dictionary. Everything happens behind the scenes.
    classifier = EMGClassifier(None)
    classifier.model = model
    # get the classifier's predictions on the test set
    preds = classifier.run(test_windows)
    om = OfflineMetrics()
    metrics = ['CA','AER','INS','REJ_RATE','CONF_MAT','RECALL','PREC','F1']
    results = om.extract_offline_metrics(metrics, test_metadata['classes'], preds[0], null_label=2)
    for key in results:
        print(f"{key}: {results[key]}")

    # and conviniently, you can access everything from the training process here
    # model.log -> has training loss, accuracy, validation loss, accuracy for every batch


    # We could also train a model with bells and whistles (rejection, velocity control, majority vote):
    # We just need to pass the training windows and training labels to the fit function or velocity control
    feature_dictionary = {}
    feature_dictionary["training_windows"] = train_windows
    feature_dictionary["train_labels"]     = train_metadata["classes"]
    classifier = EMGClassifier(None)
    classifier.add_majority_vote(3)
    classifier.add_rejection(0.9)
    classifier.add_velocity(train_windows, train_metadata["classes"])
    dl_dictionary = {"learning_rate": 1e-4,
                     "num_epochs": 50,
                     "verbose": False}
    # reset the model weights
    model = CNN(n_output   = np.unique(np.vstack(not_test_data.classes[:])).shape[0],
                n_channels = train_windows.shape[1],
                n_samples  = train_windows.shape[2],
                n_filters  = 64)
    model.fit(dataloader_dictionary, **dl_dictionary)
    classifier.model = model
     # get the classifier's predictions on the test set
    preds = classifier.run(test_windows)
    om = OfflineMetrics()
    metrics = ['CA','AER','INS','REJ_RATE','CONF_MAT','RECALL','PREC','F1']
    results = om.extract_offline_metrics(metrics, test_metadata['classes'], preds[0], null_label=2)
    for key in results:
        print(f"{key}: {results[key]}")

    

if __name__ == "__main__":
    main()