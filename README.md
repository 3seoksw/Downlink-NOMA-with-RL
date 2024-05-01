> [!NOTE]
> Currently in progress of development!

# Downlink Non-Orthogonal Multiple Access (NOMA) with Reinforcement Learning (RL)

This project is built based on the following paper: [[1]](#1).

In recent years, the Non-Orthogonal Multiple Access (NOMA) system has been considered as a promising candidate for a multiple access framework due to its performance allowing multiple users to access to channels at the same time.
Nevertheless, NOMA system has few limitations since the problem of allocating the resources (i.e. channels and powers) to users is considered to be [NP-hard](https://en.wikipedia.org/wiki/NP-hardness).

In order to mitigate this, [[1]](#1) employed an optimized power allocation method [[2]](#2) finding a sub-optimal power and proposed a reinforcement learning (RL) framework, attention-based neural network (ANN), performing channel assignment.

For further information regarding the NOMA system, please see [`docs/README.md`](https://github.com/3seoksw/Downlink-NOMA-with-RL/blob/main/docs/README.md) or [[1]](#1).

## To Get Started

### Installation

```shell
conda create --name <environment> python=3.12.2
```

Replace `<environment>` with the name of virtual environment to your liking.

Then install the required packages as such:

```shell
conda activate <environment>
pip install -r requirements.txt
```

### Training

We used [Hydra](https://github.com/facebookresearch/hydra) as a configuration tool for setting the training hyperparameters.
The configuration file can be found at `config/train.yaml`.
Simply run the following command for traing the model, and if you want to test out with different training setting, modify the `config/train.yaml` file.

```shell
python src/train.py
```

## To-Do List

- [x] ANN model
- [ ] Linear model
- [ ] CNN model
- [ ] Environment
- [ ] Base Station
- [ ] User

<a id="1" href="https://ieeexplore.ieee.org/abstract/document/8790780">[1]</a>
He, Chaofan, Yang Hu, Yan Chen, and Bing Zeng. "Joint power allocation and channel assignment for NOMA with deep reinforcement learning."
<i>IEEE Journal on Selected Areas in Communications 37</i>, no. 10 (2019): 2200-2210.

<a id="2" href="https://ieeexplore.ieee.org/abstract/document/7982784">[2]</a>
Zhu, Jianyue, Jiaheng Wang, Yongming Huang, Shiwen He, Xiaohu You, and Luxi Yang. "On optimal power allocation for downlink non-orthogonal multiple access systems."
<i>IEEE Journal on Selected Areas in Communications 35</i>, no. 12 (2017): 2744-2757.