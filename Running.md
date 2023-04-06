# Replicating the Results of this Project
First, we would recommend, using the environmen.yml.
```note
If you have a cuda enabled gpu there is environment_cuda.yml,
this is massively preffered as the experiment suite takes ~50 hours on RTX 3090.
```
To create the environment open the terminal on a Unix-like system;
```bash
conda env create -f environment<optional_cuda>.yml
```

# Logging into Weights and Biases
To run the scripts, you will need to make a Weights and Biases account. On 
wandb.ai.

After creating the environment and account, you can log into Weights and Biases with,
```bash
wandb login
```

## Project structure
```
Plotting scripts to output figures used in the reports.
*_scripts.py

run_experiments.py: Runs all experiments

eval_data/
    |__dmt_loss.npy: The IoU of the DMT with varying label proportion 
        on the test set. Sorted by label proportion.

    |__dmt_loss_dms.npy: The IoU of DMT with varying 
        difference maximised smapling on the test set. 
        Sorted by dms proportion.
        
    |__dmt_loss_epoch.npy: The IoU of DMT with varying epochs on the 
        test set. Sorted by epoch.
        
    |__plabel_loss.npy: The IoU of PLabel with the default hyperparameters
        repeated 5 times.

    |__plabel_loss_label.npy: The IoU of PLabel with varying label 
        proportion on the test set. Sorted by label proportion.

    |__baseline.npy: The IoU of the baseline with the default hyperparameters
         repeated 5 times, Sorted varying label proportion.

final_figs/
    Figures folder.

src/
|__experiments/
        |__experiment.py
            This is the main script for interaction with the code. 
            It defines a base class that implements default 
            hyperparameters and a wrapper for models.

|__models/
        The implementation of torch.nn.Modules,
        and their training and evaluation methods.

        |__UNet.py
        |__DMT.py
        |__PLabel.py
|__pet_3/
        |__data.py
            The main interface for data is defined here, 
            defining torch datasets, with other convenience methods.

        |__download_utils.py
|__utils/
        |__datasets.py
        |__evaluation.py
        |__loading.py
        |__mixin.py
        |__training.py
|__plotting/
        |__temporaty_plotting_utils.py

```

