import numpy as np


def apply_cutmix(images, labels, cutmix_fn, alpha=1.0, p=0.5):
    """
    Apply CutMix with probability p.

    Args:
        images: Batch of training images.
        labels: Corresponding class labels.
        cutmix_fn: Our existing CutMix implementation.
        alpha: Beta distribution parameter.
        p: Probability of applying CutMix.

    Returns:
        images, labels_a, labels_b, lam
    """

    if np.random.rand() < p:
        mixed_images, labels_a, labels_b, lam = cutmix_fn(
            images,
            labels,
            alpha=alpha
        )

        return mixed_images, labels_a, labels_b, lam

    # No CutMix: perform normal training
    return images, labels, labels, 1.0