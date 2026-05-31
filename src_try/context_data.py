"""Context-set construction for the deferral models.

Two things live here:

* :class:`ContextSampler` — holds the per-expert context/query tensors and
  samples random context points each training/validation step.
* :func:`get_context_data` — splits the training set into a small, fixed
  per-class *context* set and the remaining *query* set (seeded, reproducible).

The expert-simulation / prototypicality helpers that used to also live in this
module are defined in ``generate_exp_preds.py``; the copies here were dead and
have been removed.
"""

import random
from collections import defaultdict

import numpy as np
import torch

from attrdict import AttrDict


class ContextSampler():
    """Stores context and query data for every expert and samples context points.

    For ``n_experts`` experts the context/query images are broadcast to a leading
    expert dimension and paired with each expert's predictions (``mc``). Calling
    :meth:`sample` draws ``n_cntx_pts`` random context examples (shared across
    experts) to feed the rejector during a step.
    """

    def __init__(self, config, context_images, context_labels, query_images, query_labels, experts_lst, n_cntx_pts, context_expert_preds, query_expert_preds, seed=None, device='cpu', **kwargs):
        self.n_cntx_pts = n_cntx_pts
        self.device = device
        self.num_data = len(context_labels)
        self.seed = seed
        self.config = config
        self.dataset = config.dataset
        if self.dataset == 'ham10000':
            self.CLASS_NAMES = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

        self.context_images = context_images
        self.context_labels = context_labels
        self.context_expert_preds = context_expert_preds

        self.query_images = query_images
        self.query_labels = query_labels
        self.query_expert_preds = query_expert_preds

        self.experts_lst = experts_lst
        self.n_experts = len(experts_lst)

        # Convert numpy arrays to tensors if needed
        if not torch.is_tensor(self.context_labels):
            self.context_labels = torch.tensor(self.context_labels)
        if not torch.is_tensor(self.query_labels):
            self.query_labels = torch.tensor(self.query_labels)
        if not torch.is_tensor(self.context_images):
            self.context_images = torch.tensor(self.context_images)
        if not torch.is_tensor(self.query_images):
            self.query_images = torch.tensor(self.query_images)

        # Broadcast context/query images and labels over the expert dimension.
        self.cntx = AttrDict()
        self.cntx.xc = self.context_images.unsqueeze(0).repeat(self.n_experts, 1, 1, 1, 1)
        self.cntx.yc = self.context_labels.unsqueeze(0).repeat(self.n_experts, 1)

        self.cntx.mc = self.context_expert_preds
        if self.n_experts == 1:
            self.cntx.mc = self.cntx.mc.unsqueeze(0)

        self.query = AttrDict()
        self.query.xc = self.query_images.unsqueeze(0).repeat(self.n_experts, 1, 1, 1, 1)
        self.query.yc = self.query_labels.unsqueeze(0).repeat(self.n_experts, 1)

        self.query.mc = self.query_expert_preds
        if self.n_experts == 1:
            self.query.mc = self.query.mc.unsqueeze(0)

    def sample(self):
        """Return a random subset of ``n_cntx_pts`` context points (same indices for all experts)."""
        random_idxs = random.sample(range(self.num_data), self.n_cntx_pts)
        sampled_cntx = AttrDict()
        sampled_cntx.xc = self.cntx.xc[:, random_idxs, :, :, :]
        sampled_cntx.yc = self.cntx.yc[:, random_idxs]
        sampled_cntx.mc = self.cntx.mc[:, random_idxs]

        return sampled_cntx

    def send_context_to_device(self):
        """Move the full (unsampled) context tensors onto ``self.device``."""
        self.cntx.xc = self.cntx.xc.to(self.device)
        self.cntx.yc = self.cntx.yc.to(self.device)
        self.cntx.mc = self.cntx.mc.to(self.device)


def get_context_data(config, train_data, val_data, test_data=None):
    """Split ``train_data`` into a seeded per-class context set and a query set.

    A fixed ``samples_per_class`` examples per class are drawn (seeded by
    ``config["seed"]``) to form the context set; the rest become the training
    query set. Validation/test data are used whole (no split). When
    ``test_data`` is given, the test split is returned in place of validation.

    Returns ``(X_c, y_c, X_tr_q, y_tr_q, X_eval_q, y_eval_q, (cntx_indices, query_indices))``.
    """
    # the only random splitting we do for ham is on training data into query + context (dependent on seed of experiment)
    # (not for valid or test)
    samples_per_class = 15
    class_indices = defaultdict(list)
    for idx, target in enumerate(train_data.targets):
        class_indices[target.item()].append(idx)
    generator = torch.Generator().manual_seed(config["seed"])

    sampled_indices = []
    for class_label, indices in class_indices.items():
        shuffled_indices = torch.randperm(len(indices), generator=generator)[:samples_per_class]
        selected_indices = [indices[i] for i in shuffled_indices]
        sampled_indices.extend(selected_indices)

    all_indices = np.arange(len(train_data))
    query_indices = torch.tensor(list(set(all_indices) - set(sampled_indices)))
    cntx_indices = torch.tensor(sampled_indices)
    assert len(cntx_indices) + len(query_indices) == len(all_indices)
    assert set(query_indices.numpy()) | set(cntx_indices.numpy()) == set(all_indices)
    X_c = train_data.data[cntx_indices]
    y_c = train_data.targets[cntx_indices]

    X_tr_q = train_data.data[query_indices]
    y_tr_q = train_data.targets[query_indices]

    X_val_q = val_data.data
    y_val_q = val_data.targets

    if test_data is None:
        return X_c, y_c, X_tr_q, y_tr_q, X_val_q, y_val_q, (cntx_indices, query_indices)
    else:
        X_tst_q = test_data.data
        y_tst_q = test_data.targets
        return X_c, y_c, X_tr_q, y_tr_q, X_tst_q, y_tst_q, (cntx_indices, query_indices)
