"""Synthetic expert simulation used by the (non-instance-dependent) baselines.

``SyntheticExpertOverlap`` models an expert that is reliable on its specialty
classes (``classes_oracle``) with probability ``p_in`` and on every other class
with probability ``p_out``; when unreliable it emits a uniformly random label.
``simulate_experts`` partitions the classes into in-distribution (seen during
training) and out-of-distribution (held-out) experts.
"""

import random
import numpy as np


class SyntheticExpertOverlap():
    """An expert that is accurate on ``classes_oracle`` (prob. ``p_in``) and elsewhere (prob. ``p_out``).

    When the expert is "wrong" it returns a uniformly random class in
    ``[0, n_classes)`` (which may coincidentally still be correct).
    """

    def __init__(self, classes_oracle, n_classes=10, p_in=1.0, p_out=0.1):
        self.expert_static = True
        self.classes_oracle = classes_oracle
        if isinstance(self.classes_oracle, int):
            self.classes_oracle = [self.classes_oracle]
        self.n_classes = n_classes
        self.p_in = p_in
        self.p_out = p_out

    def __call__(self, labels, seed=None):
        """Return a list of predicted labels, one per entry in ``labels``."""
        if seed is not None:
            np_random_state = np.random.RandomState(seed)
            random_state = random.Random(seed)
        else:
            np_random_state = np.random
            random_state = random

        batch_size = labels.size()[0]
        predictions = [0] * batch_size
        for i in range(batch_size):
            # Reliability depends on whether this label is one of the expert's specialties.
            is_specialty = labels[i].item() in self.classes_oracle
            p_correct = self.p_in if is_specialty else self.p_out
            if np_random_state.binomial(1, p_correct) == 1:
                predictions[i] = labels[i].item()
            else:
                predictions[i] = random_state.randint(0, self.n_classes - 1)

        return predictions


def simulate_experts(config, verbose=False):
    """Build the in-distribution and out-of-distribution expert pools.

    ``n_ID_experts`` specialty classes are sampled (seeded by ``config["seed"]``)
    to define the in-distribution experts; the remaining classes define the
    out-of-distribution experts. Returns ``(experts_id, experts_ood)``.
    """
    seed = config["seed"]
    random.seed(seed)

    experts_id = []
    experts_ood = []

    sampled_ID_experts = random.sample(range(config['n_classes']), config['n_ID_experts'])
    OOD_experts = [num for num in range(config["n_classes"]) if num not in sampled_ID_experts]

    for class_oracle in sampled_ID_experts:
        expert = SyntheticExpertOverlap(classes_oracle=class_oracle, n_classes=config["n_classes"], p_in=config['p_in'], p_out=config['p_out'])
        experts_id.append(expert)

    for class_oracle in OOD_experts:
        expert = SyntheticExpertOverlap(classes_oracle=class_oracle, n_classes=config["n_classes"], p_in=config['p_in'], p_out=config['p_out'])
        experts_ood.append(expert)

    if verbose:
        print(f"In-distribution experts: {[i.classes_oracle[0] for i in experts_id]}")
        print(f"Out-of-distribution experts: {[i.classes_oracle[0] for i in experts_ood]}")

    return experts_id, experts_ood
