# Edited by Dongyu Zhang
import random
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, mean_squared_error
import datasets
from transformers import Trainer
from torch.utils.data.dataset import Subset


class ddict(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def random_split_dataset(dataset, lengths, random_seed=20200706):
    # state snapshot
    state = {}
    state['seeds'] = {
        'python_state': random.getstate(),
        'numpy_state': np.random.get_state(),
        'torch_state': torch.get_rng_state(),
        'cuda_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    }

    # seed
    random.seed(random_seed)  # python
    np.random.seed(random_seed)  # numpy
    torch.manual_seed(random_seed)  # torch
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)  # torch.cuda

    dataset_list = torch.utils.data.dataset.random_split(dataset, lengths)

    # reinstate state
    random.setstate(state['seeds']['python_state'])
    np.random.set_state(state['seeds']['numpy_state'])
    torch.set_rng_state(state['seeds']['torch_state'])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(state['seeds']['cuda_state'])

    return dataset_list


def ordered_split_dataset(dataset, lengths):
    start = 0
    dataset_list = []
    for lN in lengths:
        end = start + lN
        new_indices = list(range(start, end))
        dataset_list.append(Subset(dataset=dataset, indices=new_indices))
        start = end

    return dataset_list


def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def compute_cls_metrics(eval_pred):
    m = torch.nn.Softmax(dim=1)
    predictions, labels = eval_pred
    probs = m(torch.Tensor(predictions))
    _, y_pred = torch.max(probs, 1)
    y_values = probs[:, 1]
    f_s = f1_score(labels, y_pred)
    p_s = precision_score(labels, y_pred)
    r_s = recall_score(labels, y_pred)
    try:
        auc_s = roc_auc_score(labels, y_values)
    except ValueError:
        auc_s = 0.0
    return {"f1_score": float(f_s), "precision_score": float(p_s), "recall_score": float(r_s),
            "auc_score": float(auc_s)}


def compute_reg_metrics(eval_pred):
    predictions, labels = eval_pred
    mse = mean_squared_error(labels, predictions)
    return {"mse_score": float(mse)}

# class GetEmbedsTrainer(Trainer):
#     def get_all_embeds(self, model, inputs):
#         labels = inputs.get("labels")
#         # forward pass
#         outputs = model(**inputs)
#         return (labels,) + outputs

