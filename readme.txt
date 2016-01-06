We are trying to develop the patient status recognition with
deep learning.
We' d like to share the source code.
We use pylearn2, a machine learning library.
Our system is as following
 Ubuntu14.04LTS
 Python 2.7
 python-opencv 2.4.8
 Theano 0.7
 pylearn2
 cuda-7-0

There are the source code and dataset files.
traincnn.yaml: configuration for training a model.
ds.py: dataset class of the patient status recognition for pylearn2.
ds.pkl.gz: the compressed dataset.

To train the model, run the following command.
$ python traincnn.py
The trained model will be saved as "cnn_best.pkl".

To see the performance of the model, run the following command.
$ print_monitor.py cnn_best.pkl|grep test_y_misclass
"print_monitor.py" is in "pylearn2/scipts/".
The evaluation is in frame level.

The DNN model(defined in the YAML file) contails many hyperparameters. 
We have to optimize the hyperparameters.

To make "ds.pkl.gz", put videos and the annotations to dataset/original/
and run the following command.
$ cd dataset
$ ./resize.sh
$ cd ../
$ python mkdspkl.py --train train.txt --test test.txt ds.pkl.gz

