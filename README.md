# Fundamentals of inference and learning project

### Environment setup

* Create a conda environment using pythonb 3.10: `conda create -n fil_project python=3.10`
* Activate the environment: `conda activate fil_project`
* Move to the projects'directory `cd FIL_project`
* Execute the command `pip install -r requirements.txt`


### Structure of the files:

#### MNIST/
Contains the original data set from www.di.ens.fr/~lelarge/MNIST.tar.gz
It contains 60000 images and a test dataset of 10000 images. They are in 'MNIST/processed/training.pt' and 'MNIST/processed/test.pt'.

#### create_data_set.ipynb
The goal of this notebook is to generate a training dataset of 4000 images, by subsampling the original training dataset. And generate a testing dataset of 4000 images, by subsampling the original testing dataset.

#### dataset/
Contains the subsampled train and test datasets

#### create_parameter_arrays.py
The goal of this file is to create the arrays N_param and H.
N_param contains the number of parameters of all the neural networks to obtain figure 4.
H contains the hidden layer size corresponding to the number of parameters in the array N_param. Remark: H does not exactly produce a number of parameters equal to the ones in N_param because there is a rounding step after applying the formula to go from number of parameters to hidden size.

#### parameters_arrays/
This folder contains the arrays N_param and H.

#### main.py
This code allows to automate the training of all the models needed to reproduce the figure 4 of the paper. It allows to do multiple runs. 1 run = 1 figure 4 of the paper without average. The purpose of this is to be able to do an average between all the runs and obtain an averaged figure 4. (for this see plot_figure.ipynb)

This is the main file of the project.

It goes trough all the values in N_param and H to do 1 run. And it uses the subsampled datasets from dataset/

On the terminal this code displays te progression of the trainings within each run.

#### train_helpers.py
This code contains the logic to train one model within one run. Used by main.py.

#### helpers.py
This code contains definitions of classes and function necessary to train a model. Used by train_helpers.py.

#### runs/
This folder contains the outputs of main.py. It is the training data of all the trained models within one run, for all runs.

#### runs_saved/
Just a place to store completed runs. To avoid having them overridden when launching main.py.
It contains the two complete runs done to produce the figure 4 presented in the project report. It also contains the cmd prompt output when running main.py which produced the two runs. But the cmd prompt output is not the most recent one in terms of user interface. Run the code to see the latest uer interface.

#### monitor_training.ipynb
This notebook allows to look at the data produced when models are trained with the main.py program.
It allows to monitor in real time the training of the models when we run main.py.
It is also useful because it allows to look at the training data of any trained models when the training is done. 

#### plot_figure.ipynb
The goal of this notebook is to plot the figure 4 of the paper. Averaged between runs or per run.

### Remarks:
All the codes above have been used with the cuda version of Pytorch (installed with requirements.txt) with a RTX4060. On a laptop with Windows 11.

I didn't use requirement.txt to do the pytorch installation, I did an installation with pip install with a command from the Pytorch website. Weirdly now that I use requirements.txt I am not able to use cuda with main.py so this may be also a problem that occurs on your machine. So it is limited to using the cpu.

With the provided environment all the codes can be re-run without any problem except the problem mentioned above (limitation to use the cpu instead of a graphics card).