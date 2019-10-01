# MeLU: Meta-Learned User Preference Estimator for Cold-Start Recommendation

PyTorch implementation of the paper: "MeLU: Meta-Learned User Preference Estimator for Cold-Start Recommendation", KDD, 2019.

## Abstract
This paper proposes a recommender system to alleviate the coldstart problem that can estimate user preferences based on only a small number of items. To identify a user’s preference in the cold state, existing recommender systems, such as Netflix, initially provide items to a user; we call those items evidence candidates. Recommendations are then made based on the items selected by the user. Previous recommendation studies have two limitations: (1) the users who consumed a few items have poor recommendations and (2) inadequate evidence candidates are used to identify user preferences. We propose a meta-learning-based recommender system called MeLU to overcome these two limitations. From metalearning, which can rapidly adopt new task with a few examples, MeLU can estimate new user’s preferences with a few consumed items. In addition, we provide an evidence candidate selection strategy that determines distinguishing items for customized preference estimation. We validate MeLU with two benchmark datasets, and the proposed model reduces at least 5.92% mean absolute error than two comparative models on the datasets. We also conduct a user study experiment to verify the evidence selection strategy.

## Usage
### Requirements
- python 3.6+
- pytorch 1.1+
- tqdm 4.32+
- pandas 0.24+

### Preparing dataset
It needs about 22GB of hard disk space.
```python
import os
from data_generation import generate
master_path= "./ml"
if not os.path.exists("{}/".format(master_path)):
    os.mkdir("{}/".format(master_path))
    generate(master_path)
```

### Training a model
Our model needs support and query sets. The support set is for local update, and the query set is for global update.
```python
import torch
import pickle
from MeLU import MeLU
from options import config
from model_training import training
melu = MeLU(config)
model_filename = "{}/models.pkl".format(master_path)
if not os.path.exists(model_filename):
    # Load training dataset.
    training_set_size = int(len(os.listdir("{}/warm_state".format(master_path))) / 4)
    supp_xs_s = []
    supp_ys_s = []
    query_xs_s = []
    query_ys_s = []
    for idx in range(training_set_size):
        supp_xs_s.append(pickle.load(open("{}/warm_state/supp_x_{}.pkl".format(master_path, idx), "rb")))
        supp_ys_s.append(pickle.load(open("{}/warm_state/supp_y_{}.pkl".format(master_path, idx), "rb")))
        query_xs_s.append(pickle.load(open("{}/warm_state/query_x_{}.pkl".format(master_path, idx), "rb")))
        query_ys_s.append(pickle.load(open("{}/warm_state/query_y_{}.pkl".format(master_path, idx), "rb")))
    total_dataset = list(zip(supp_xs_s, supp_ys_s, query_xs_s, query_ys_s))
    del(supp_xs_s, supp_ys_s, query_xs_s, query_ys_s)
    training(melu, total_dataset, batch_size=config['batch_size'], num_epoch=config['num_epoch'], model_save=True, model_filename=model_filename)
else:
    trained_state_dict = torch.load(model_filename)
    melu.load_state_dict(trained_state_dict)
```

### Extracting evidence candidates
We extract evidence candidate list based on the MeLU.
```python
from evidence_candidate import selection
evidence_candidate_list = selection(melu, master_path, config['num_candidate'])
for movie, score in evidence_candidate_list:
    print(movie, score)
```
Note that, you may have a different evidence candidate list from the paper. That's because we do not open the random seeds of data generation and model training.

## Citation
If you use this code, please cite the paper.
```
@inproceedings{lee2019melu,
  title={MeLU: Meta-Learned User Preference Estimator for Cold-Start Recommendation},
  author={Lee, Hoyeop and Im, Jinbae and Jang, Seongwon and Cho, Hyunsouk and Chung, Sehee},
  booktitle={Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={1073--1082},
  year={2019},
  organization={ACM}
}
```