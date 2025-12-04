# CBExplorer
Implementation of Concept Bottleneck Explorer for ECS289H

## Model

## Interface

# Run Backend 

- First, install dependencies.

```
conda create -n posthocgencbm python=3.8
conda install nvidia/label/cuda-11.7.0::cuda-nvcc cudatoolkit
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```
- Then, run the server by 

```
cd <Path_To_CBExplorer>
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```