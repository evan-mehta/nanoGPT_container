#!/bin/bash


#podman run -it -v /Users/evanmehta/Downloads/ngpt.sh:/tmp/ngpt.sh ubuntu bash /tmp/ngpt.sh

export DEBIAN_FRONTEND=noninteractive

apt update
apt install -y python3 pip git

git clone https://github.com/karpathy/nanoGPT.git

cd nanoGPT/

apt install -y python3.12-venv

python3 -m venv my_venv 

. my_venv/bin/activate

pip install torch numpy transformers datasets tiktoken wandb tqdm

python3 data/shakespeare_char/prepare.py

python3 train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0

python3 sample.py \
    --start="What is ten plus ten?" \
    --num_samples=5 --max_new_tokens=100 --out_dir=out-shakespeare-char --device=cpu
