## Setup
```
conda create -n multi-irl python=3.11 pip
conda activate multi-irl
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```
## To Run
First, download the D4RL data locally (make sure to update `expert_data_path` in `config.yaml` if necessary).
```
conda activate multi-irl
export PYTHONPATH=${pwd}:$PYTHONPATH
python scripts/train_irl.py
```
