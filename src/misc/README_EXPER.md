# Guide on how to run experiments for DMT


### Architecture for Experiments

The architecture of the experiments is dependent on classes. Any experiment performed in this project are subclasses of a parent class BaseExperiments.  
  
The BaseExperiment class contain all the hyper-parameters needed for models within Dynamic Mutual Training (DMT) and for the Baseline models.
These are set to some default values that needs to be overridden for experiments to reflect changes in the experiment conducted. 

After all experiment subclasses have been created, they are all registered in a global class Experiments that keeps a registry on all the experiments that will be conducted (i.e., all the subclasses). If the user then wants to run all the experiments, the Experiment class have a functionality for this that will be explained below. However, without proper GPU and memory this will be tedious. Even in the case for running a single experiment for a subclass, the user will rely on having signficant computing power (robust GPU).

### How the experiments are created
First, we have the Experiment class

```python 
class Experiments:
    def register:
        # keeps a experiment registry from Base Experiments

    def run_all():
        # runs all registered experiments
```

Then, we have the BaseExperiment class

```python 
class BaseExperiment(ABC):
    # initialize and holds all hyper-parameters used for
    # running a DMT/Baseline model

    # the properties of each experiment
    @property
    def description():
        # add model description

    @property
    def model_folder():
        # model folder to save experiment(s)
    
    # this is the function to override for running 
    # constructed experiments
    @abstractmethod
    def run():
        # run some of the below run types
    
    # the type of runs to do for an experiment
    def _train_baseline_only(**baseline_parameters)
        # trains the baseline model and saves it
    
    def _base_run(**dmt_parameters):
        # trains and saves dmt model
    
    def _plabel_run(**plabel_parameters):
        # trains model on a varying proportion of 
        # labeled data with pseudo labels in training after 
        # pre-training
```
Now, for the base experiment, we have three run cases. 
These are all different with respect to what your experiment is performing, and they can all be used in combination with each other. The functionality of the seperate run functions are:
-  **_plabel_run**: Pseudo-label run method for all experiments. This method is called for pseudo label experiments for subclasses. This method trains the PLabel model and saves the best model (See PLabel class for details). If a baseline model is provided, it is also trained and saved.
- **_base_run**: Base run method for all experiments. This method is called by the subclass run method. This method trains the DMT model and saves the best model.
- **_train_baseline_only**: Training the baseline model only. Used for comparison with DMT.


Now, to construct an experiment, what you do is:
```python
class MyExperiment(BaseExperiments):
    # create some hyperparameters
    # in here that you want to test

    @property
    def model_folder():
        # set a return string with model folder
    
    @property 
    def description():
        # set a return description of the experiment
    def run():
        # Implement the experiment performed here
        # inside use the run function from the three cases
        # of run experiment that you want to use
```
### How to run experiments

To perform DMT / Baseline experiments in the shell, first navigate:

``` path/to/dir/src/experiments``` 

Then, if you want to want to conduct your own experiment you can first import Baseline as followed:

```
python from /path/to/dir/src.experiments import BaseExperiments
```

Then follow the steps above for creating an experiment, and then do:
```
my_experiment = MyExperiment()
my_experiment.run()
```

### Our Experiments

We have the following list of conducted experiments in this project:
- **TrainBaselines**
- **VaryDifferenceMaximization**
- **VaryDMTEpochs**
- **VaryLabelProportion**
- **PLabelVaryLabelProportion**
- **PlabelDefault**

To conduct any of them follow the steps from the "**How to run experiments**". If you want to conduct all of them at once, do:
```
/path/to/dir/src/experiments python run_experiments.py
```


