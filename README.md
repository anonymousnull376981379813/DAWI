# DAWI: Dual Anchor Weighted Interpolation for Unlearning
This is the official repository for DAWI. Code was tested on an RTX 4090, but any Nvidia GPU with over 24GB of VRAM should work. Python3 version used was 3.11

To get started with TOFU, run
```
python3.11 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
model=Llama-3.2-1B-Instruct
cd open-unlearning
python setup_data.py --eval
cd ..
```

To reproduce Table 1 of the paper, run
```
python entrypoint.py --mode=figure1 --dataset=TOFU
```

To evaluate a model on TOFU, run the following. Paths for evaluations should be absolute paths. Evaluations will appear in the evals folder. 
```
python entrypoint.py --mode=eval --dataset=TOFU --model_name_or_path=/absolute/path/to/model
```

To train on TOFU, run 
```
python entrypoint.py --mode=train --dataset=TOFU --save_path=TEST
```

To get started with MUD, run 
```
python3.10 -m venv mud_venv
source mud_venv/bin/activate
pip3 install -r mud_requirements.txt
```

To evaluate MUD with DAWI, run 
```
python3 entrypoint.py --mode=eval  --dataset=MUD20k --model_name_or_path=Anonymous0192830/DAWI_MUD20k
```
Results are under mud20k