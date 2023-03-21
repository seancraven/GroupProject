DMT:
 - [x] add load_state_dict() to UNET.py
 - [ ] Batch problem fix, currently we dont iterate through all of the unlabeled training examples.
 - [ ] Pretrain runs untill convergence on validation set
 - [ ] validation set refactor.
 - [ ] Per teacher student DMT paper figure 6. 
 - [ ] Baseline validation, save model.

Pet Dataset:
 - [ ] Add validation split function
 - [ ] create more finegrain dataset splits. [0.005, ..., 0.15]

Plotting from wandb:
 - [ ] Learning graphs.

Model testing utils
 - [ ] Currently doesn't work at all, get working.

Research Question:
 - [ ] Gamma sweep? Looking at gamma ranges to see where it fails? 
 - [ ] Ablation study fixed vs decay gamma. 
