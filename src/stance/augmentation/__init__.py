"""Training-time augmentations (Phase 6). Each function maps a DataFrame of train rows to a
DataFrame of augmented copies with the same labels and group_ids."""
from .noise import augment_frame, char_noise, drop_matras, repeat_chars  # noqa: F401
