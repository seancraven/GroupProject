DMT: Michael
 - [x] add load_state_dict() to UNET.py
 - [x] Batch problem fix, currently we dont iterate through all of the unlabeled training examples.
 - [x] Pretrain runs untill convergence on validation set
 - [x] validation set refactor.
 - [ ] Per teacher student DMT paper figure 6.
 - [ ] Baseline validation, save model.
 - [ ] Code Review: Daniel , Alexi, Eirik
 - [ ] Model saving protocol to tell Max.

Pet Dataset:
 - [x] Add validation split function, Tristan, Jannette, Charlie
 - [ ] Test validation split function, Tristan, Jannette, Charlie

```python
class Pets(Dataset):
    def __init__(self, ........):
    some shit.

    def validation_split(self, split_fraction: float) -> Tuple[Pets, Pets]:
        return

```
 - [x] create more finegrain dataset splits. [0.005, ..., 0.15]

Plotting from wandb:
 - [ ] Learning graphs.: Charlie

Model testing utils
 - [ ] Currently doesn't work at all, get working.: Max

Research Question:
 - [ ] Gamma sweep? Looking at gamma ranges to see where it fails?
 - [ ] Ablation study fixed vs decay gamma.

Readmes
 - [ ] Datasets: Jannette, Charlie, Tristan
 - [ ] DMT: Eirik, Alexei, Daniel


 - [x] Label Proportioins Need to be decided [0.01, 0.02, 0.05, .1, .5, 0.1]: Sean

Bossman:
 - [ ] Alexei.

Binman:
 - [x] Sean, clean out repo of crap.

Other Sean Activities:
 - [ ] Make validation set same size as labelled set.
 - [ ] Train models.
 - [ ] quick and dirty plotting utils so that we can demostrate something.
 - [ ] depracate the old dataset interface.
 - [ ] Add validation from file.
