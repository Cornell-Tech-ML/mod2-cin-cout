[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/YFgwt0yY)
# MiniTorch Module 2

<img src="https://minitorch.github.io/minitorch.svg" width="50%">


* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module2/module2/

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py project/run_manual.py project/run_scalar.py project/datasets.py


## Task 2.5: Training
### TODO

Implement a neural network over the data with three linears (2-> Hidden (relu), Hidden -> Hidden (relu), Hidden -> Output (sigmoid)). It should do exactly the same thing as the corresponding functions in project/run_scalar.py, but now use the tensor code base.

Train a tensor model and add your results for all datasets to the README.

Record the time per epoch reported by the trainer. (It is okay if it is slow).


### Simple
#### DataSet
![SimpleDataSet](images/SimpleDataset.PNG)
#### Model
![SimpleModel](images/SimpleModel.PNG)
#### Hyperparameters and Final Images
![SimpleHyperparameters](images/SimpleHyperparameters.PNG)
#### Training Logs
![SimpleLog](images/SimpleResult.PNG)

### Diag
#### DataSet
![DiagDataSet](images/DiagDataset.PNG)
#### Model
![DiagModel](images/DiagModel.PNG)
#### Hyperparameters and Final Images
![DiagHyperparameters](images/DiagHyperparameters.PNG)
#### Training Logs
![DiagLog](images/DiagResult.PNG)

### Split
#### DataSet
![splitDataSet](images/SplitDataset.PNG)
#### Model
![splitModel](images/SplitModel.PNG)
#### Hyperparameters and Final Images
![splitHyperparameters](images/SplitHyperparameters.PNG)
#### Training Logs
![splitLog](images/SplitResult.PNG)

### Xor
#### DataSet
![xorDataSet](images/XorDataset.PNG)
#### Model
![xorModel](images/XorModel.PNG)
#### Hyperparameters and Final Images
![xorHyperparameters](images/XorHyperparameters.PNG)
#### Training Logs
![xorLog](images/XorResult.PNG)

