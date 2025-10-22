# 😀😀😀

## Quick Start
```
git clone https://github.com/minsuk00/SingleImageINR.git
cd SingleImageINR

# create env
conda create -n sio python=3.10 -y
conda activate sio

# dependencies
module load cuda/12.1.1 gcc/11.2.0
pip install --upgrade pip
pip install -r requirements.txt

# download CheSS weight and place in project root
# https://github.com/mi2rl/CheSS
```