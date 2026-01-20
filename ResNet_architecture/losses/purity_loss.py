import tensorflow as tf


class PurityLoss:
    """
    Prototype Purity Loss.

    Encourages each prototype to activate strongly on only ONE tumor class.
    - High activation where prototype's target class exists
    - Low activation elsewhere

    Critical for establishing class-specific prototypes.
    """

    def __init__(self, n_prototypes=3):
        self.n_prototypes = n_prototypes

    def __call__(self, masks, similarities):
        """
        Compute Purity loss.

        Args:
            masks: (B, D, H, W, n_classes) one-hot ground truth masks
            similarities: (B, D, H, W, n_prototypes) prototype similarity maps

        Returns:
            Purity loss scalar
        """
        loss = 0.0

        for proto_idx in range(self.n_prototypes):
            target_class = proto_idx + 1  # proto 0 → class 1, etc.

            # Get similarity for this prototype
            proto_sim = similarities[..., proto_idx]  # (B, D, H, W)

            # Target mask (where this prototype should activate)
            target_mask = masks[..., target_class]  # (B, D, H, W)

            # Non-target mask
            non_target_mask = 1.0 - target_mask

            # High similarity where mask matches (maximize → negative)
            loss -= tf.reduce_mean(proto_sim * target_mask)

            # Low similarity elsewhere (minimize → positive)
            loss += tf.reduce_mean(proto_sim * non_target_mask)

        return loss / self.n_prototypes
