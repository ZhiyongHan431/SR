"""Explicit MACA runtime policy, based on fixed-input convolution diagnostics."""
import os


def configure_maca_numeric_policy():
    selected = os.environ.get('GRN_S5_DETERMINISTIC', '1')
    if selected not in ('0', '1'):
        raise ValueError('GRN_S5_DETERMINISTIC must be 0 or 1')
    if selected == '0':
        return
    import torch
    # The accelerated BF16/FP16 Conv2d path was non-repeatable even with
    # deterministic=True. Disabling this backend stabilized full inference.
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
