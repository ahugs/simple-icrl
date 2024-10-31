# Simplifying Constraint Inference with Inverse Reinforcement Learning


## Setup - Conda
```
conda create -n multi-irl python=3.11 pip
conda activate multi-irl
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
cd envs
pip install -e .
```
## Setup - Docker
Build and run the provided Dockerfile

## To Run 
First, download the expert data provided by Liu et al. [here](https://github.com/Guiliang/ICRL-benchmarks-public/tree/main) (make sure to update `expert_data_path` in config files if necessary).
```
conda activate multi-irl
export PYTHONPATH=${pwd}:$PYTHONPATH
python scripts/train_icrl.py env={NAME_OF_ENV}", f"expert_data_path={LOCATION_OF_EXPERT_DATA}", f"+experiment={NAME_OF_EXPERIMENT}
```

`NAME_OF_ENV` and `NAME_OF_EXPERIMENT` must correspond to one of the names of the YAML files provided in `config/env` and `config/experiment`, respectively.

To run the experiments in the paper, use the following experiment names:

- IRL-Base: `orig_irl_rn.yaml`

- IRL+L2: `l2.yaml`

- IRL+L2+SC: `sc_l2_warmstart.yaml`

- IRL+L2+PR: `l2_pr_warmstart.yaml`

- IRL+L2+SC+PR (IRL-Plus): `sc_l2_pr_warmstart.yaml`



Experiments with seperate critics should be run with `scripts/train_icrl.py`. Otherwise, should be run with `scripts/train_simple_icl.py`


