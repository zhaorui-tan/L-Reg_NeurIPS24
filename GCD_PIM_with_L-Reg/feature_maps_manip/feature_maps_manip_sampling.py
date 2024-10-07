# feature_maps_manip.py module provides the function to get the saved feature maps

import os
import numpy as np


def get_fm_preds_and_gt_labels(set_path):
    print("Loading feature map dataset...")
    with open(os.path.join(set_path, "all_feats.npy"), "rb") as f_all_feats:
        all_feats = np.load(f_all_feats)
    with open(os.path.join(set_path, "mask_lab.npy"), "rb") as f_mask_lab:
        # True for a labeled point, False for an unlabeled point
        mask_lab = np.load(f_mask_lab)
    with open(os.path.join(set_path, "mask_cls.npy"), "rb") as f_mask_cls:
        # True if the point is from an old class,
        # False if the point is from a novel class
        mask_cls = np.load(f_mask_cls)
    with open(os.path.join(set_path, "targets.npy"), "rb") as f_targets:
        targets = np.load(f_targets)
    l_feats = all_feats[mask_lab]  # Get labeled set Z_L
    u_feats = all_feats[~mask_lab]  # Get unlabeled set Z_U
    l_targets = np.asarray(targets[mask_lab], dtype=int)  # Get labeled targets
    u_targets = np.asarray(targets[~mask_lab], dtype=int)  # Get unlabeled targets

    # Get portion of mask_cls which corresponds to the unlabeled set
    mask = mask_cls[~mask_lab]
    mask = mask.astype(bool)

    nbr_of_classes = int(np.max(targets) + 1)

    return (
        np.asarray(all_feats),
        np.asarray(targets, dtype=int),
        nbr_of_classes,
        mask_cls.astype(bool),
        mask_lab.astype(bool),
    )


#  setting 6
def sample_mask_lab(mask_lab, mask_cls, targets, sampling_ratio=0.8):
    # Ensure mask_lab and mask_cls are boolean arrays
    mask_lab = mask_lab.astype(bool)
    mask_cls = mask_cls.astype(bool)

    # print(targets, np.unique(targets))
    # Filter targets to only include those from labeled points
    labeled_targets = targets[mask_lab]

    # Get unique classes from labeled points
    unique_classes = np.unique(labeled_targets)
    # print(unique_classes)
    # np.random.shuffle(unique_classes)
    # print(len(unique_classes))

    # Filter to classes that are in half of mask_cls (old or new classes)
    half_classes = (
        unique_classes[: int(len(unique_classes) * 0.5)   ]
        # if mask_cls[0]
        # else unique_classes[int(len(unique_classes) * 0.55): ]
    )
    # half_classes =unique_classes


    # print(half_classes)
    # Create a mask for labeled samples that belong to the selected half-classes
    class_mask = np.isin(labeled_targets, half_classes)

    # Get indices of labeled samples belonging to half-classes
    labeled_half_class_indices = np.where(class_mask)[0]


    # Sample 75% of these indices
    sample_size = int(class_mask.sum() * sampling_ratio)
    sampled_indices = np.random.choice(
        labeled_half_class_indices, size=sample_size, replace=False
    )

    left_sample_indices = np.setdiff1d(labeled_half_class_indices, sampled_indices, assume_unique = True)


    # Create a final mask for all samples, default to False
    final_mask = np.zeros_like(mask_lab, dtype=bool)
    final_mask[mask_lab] = np.isin(np.arange(sum(mask_lab)), sampled_indices)


    valid_mask = mask_lab.astype(int) - final_mask.astype(int)
    # print(mask_lab, final_mask, valid_mask)
    # valid_mask = valid_mask.astype(bool)


    valid_mask_seen_mask = np.zeros_like(mask_lab, dtype=bool)
    valid_mask_seen_mask[mask_lab] = np.isin(np.arange(sum(mask_lab)), left_sample_indices)
    valid_mask_unseen_mask = valid_mask.astype(int) - valid_mask_seen_mask.astype(int)
    final_mask,valid_mask,valid_mask_seen_mask,valid_mask_unseen_mask =  final_mask.astype(bool), valid_mask.astype(bool), valid_mask_seen_mask.astype(bool), valid_mask_unseen_mask.astype(bool),
    # print('valid_mask',targets[valid_mask], np.unique(targets[valid_mask]))
    # print('valid_mask_seen_mask',targets[valid_mask_seen_mask], np.unique(targets[valid_mask_seen_mask]))
    # print('valid_mask_unseen_mask', targets[valid_mask_unseen_mask], np.unique(targets[valid_mask_unseen_mask]))

    return final_mask,valid_mask,valid_mask_seen_mask,valid_mask_unseen_mask, labeled_targets

# #  setting 6
# def sample_mask_lab(mask_lab, mask_cls, targets, sampling_ratio=0.5):
#     # Ensure mask_lab and mask_cls are boolean arrays
#     mask_lab = mask_lab.astype(bool)
#     mask_cls = mask_cls.astype(bool)
#
#     # Filter targets to only include those from labeled points
#     labeled_targets = targets[mask_lab]
#
#     # Get unique classes from labeled points
#     unique_classes = np.unique(labeled_targets)
#     # print(unique_classes)
#     np.random.shuffle(unique_classes)
#     # print(len(unique_classes))
#
#     # Filter to classes that are in half of mask_cls (old or new classes)
#     half_classes = (
#         unique_classes[: int(len(unique_classes) * 0.5)   ]
#         # if mask_cls[0]
#         # else unique_classes[int(len(unique_classes) * 0.3): ]
#     )
#     # print(half_classes)
#     # half_classes =unique_classes
#
#
#     # print(half_classes)
#     # Create a mask for labeled samples that belong to the selected half-classes
#     class_mask = np.isin(labeled_targets, half_classes)
#
#     # Get indices of labeled samples belonging to half-classes
#     labeled_half_class_indices = np.where(class_mask)[0]
#
#     # Sample 75% of these indices
#     sample_size = int(class_mask.sum() * sampling_ratio)
#     sampled_indices = np.random.choice(
#         labeled_half_class_indices, size=sample_size, replace=False
#     )
#
#     left_sample_indices = np.setdiff1d(labeled_half_class_indices, sampled_indices, assume_unique = True)
#
#
#     # Create a final mask for all samples, default to False
#     final_mask = np.zeros_like(mask_lab, dtype=bool)
#     final_mask[mask_lab] = np.isin(np.arange(sum(mask_lab)), sampled_indices)
#
#     valid_mask = mask_lab.astype(int) - final_mask.astype(int)
#     # print(mask_lab, final_mask, valid_mask)
#     valid_mask = valid_mask.astype(bool)
#
#     valid_mask_seen_mask = np.zeros_like(mask_lab, dtype=bool)
#     valid_mask_seen_mask[mask_lab] = np.isin(np.arange(sum(mask_lab)), left_sample_indices)
#
#     valid_mask_unseen_mask = valid_mask.astype(int) - valid_mask_seen_mask.astype(int)
#
#
#     return final_mask, valid_mask, valid_mask_seen_mask, valid_mask_unseen_mask

# # setting 5
# def sample_mask_lab(mask_lab, mask_cls, targets, sampling_ratio=0.8):
#     # Ensure mask_lab and mask_cls are boolean arrays
#     mask_lab = mask_lab.astype(bool)
#     mask_cls = mask_cls.astype(bool)
#
#     # Filter targets to only include those from labeled points
#     labeled_targets = targets[mask_lab]
#
#     # Get unique classes from labeled points
#     unique_classes = np.unique(labeled_targets)
#     # print(unique_classes)
#     np.random.shuffle(unique_classes)
#     # print(len(unique_classes))
#
#     # Filter to classes that are in half of mask_cls (old or new classes)
#     half_classes = (
#         unique_classes[: int(len(unique_classes) * 0.3)   ]
#         if mask_cls[0]
#         else unique_classes[int(len(unique_classes) * 0.3): ]
#     )
#     # print(half_classes)
#     # half_classes =unique_classes
#
#
#     # print(half_classes)
#     # Create a mask for labeled samples that belong to the selected half-classes
#     class_mask = np.isin(labeled_targets, half_classes)
#
#     # Get indices of labeled samples belonging to half-classes
#     labeled_half_class_indices = np.where(class_mask)[0]
#
#     # Sample 75% of these indices
#     sample_size = int(class_mask.sum() * sampling_ratio)
#     sampled_indices = np.random.choice(
#         labeled_half_class_indices, size=sample_size, replace=False
#     )
#
#     left_sample_indices = np.setdiff1d(labeled_half_class_indices, sampled_indices, assume_unique = True)
#
#
#     # Create a final mask for all samples, default to False
#     final_mask = np.zeros_like(mask_lab, dtype=bool)
#     final_mask[mask_lab] = np.isin(np.arange(sum(mask_lab)), sampled_indices)
#
#     valid_mask = mask_lab.astype(int) - final_mask.astype(int)
#     # print(mask_lab, final_mask, valid_mask)
#     valid_mask = valid_mask.astype(bool)
#
#     valid_mask_seen_mask = np.zeros_like(mask_lab, dtype=bool)
#     valid_mask_seen_mask[mask_lab] = np.isin(np.arange(sum(mask_lab)), left_sample_indices)
#
#     valid_mask_unseen_mask = valid_mask.astype(int) - valid_mask_seen_mask.astype(int)
#
#
#     return final_mask, valid_mask, valid_mask_seen_mask, valid_mask_unseen_mask



# # setting 4
# def sample_mask_lab(mask_lab, mask_cls, targets, sampling_ratio=0.8):
#     # Ensure mask_lab and mask_cls are boolean arrays
#     mask_lab = mask_lab.astype(bool)
#     mask_cls = mask_cls.astype(bool)
#
#     # Filter targets to only include those from labeled points
#     labeled_targets = targets[mask_lab]
#
#     # Get unique classes from labeled points
#     unique_classes = np.unique(labeled_targets)
#     # print(unique_classes)
#     np.random.shuffle(unique_classes)
#     # print(len(unique_classes))
#
#     # Filter to classes that are in half of mask_cls (old or new classes)
#     half_classes = (
#         unique_classes[: int(len(unique_classes) * 0.25)   ]
#         if mask_cls[0]
#         else unique_classes[int(len(unique_classes) * 0.25): ]
#     )
#     # print(half_classes)
#     # half_classes =unique_classes
#
#
#     # print(half_classes)
#     # Create a mask for labeled samples that belong to the selected half-classes
#     class_mask = np.isin(labeled_targets, half_classes)
#
#     # Get indices of labeled samples belonging to half-classes
#     labeled_half_class_indices = np.where(class_mask)[0]
#
#     # Sample 75% of these indices
#     sample_size = int(class_mask.sum() * sampling_ratio)
#     sampled_indices = np.random.choice(
#         labeled_half_class_indices, size=sample_size, replace=False
#     )
#
#     left_sample_indices = np.setdiff1d(labeled_half_class_indices, sampled_indices, assume_unique = True)
#
#
#     # Create a final mask for all samples, default to False
#     final_mask = np.zeros_like(mask_lab, dtype=bool)
#     final_mask[mask_lab] = np.isin(np.arange(sum(mask_lab)), sampled_indices)
#
#     valid_mask = mask_lab.astype(int) - final_mask.astype(int)
#     # print(mask_lab, final_mask, valid_mask)
#     valid_mask = valid_mask.astype(bool)
#
#     valid_mask_seen_mask = np.zeros_like(mask_lab, dtype=bool)
#     valid_mask_seen_mask[mask_lab] = np.isin(np.arange(sum(mask_lab)), left_sample_indices)
#
#     valid_mask_unseen_mask = valid_mask.astype(int) - valid_mask_seen_mask.astype(int)
#
#
#     return final_mask, valid_mask, valid_mask_seen_mask, valid_mask_unseen_mask





# # setting 3
# def sample_mask_lab(mask_lab, mask_cls, targets, sampling_ratio=0.9):
#     # Ensure mask_lab and mask_cls are boolean arrays
#     mask_lab = mask_lab.astype(bool)
#     mask_cls = mask_cls.astype(bool)
#
#     # Filter targets to only include those from labeled points
#     labeled_targets = targets[mask_lab]
#
#     # Get unique classes from labeled points
#     unique_classes = np.unique(labeled_targets)
#     # print(unique_classes)
#     np.random.shuffle(unique_classes)
#     # print(len(unique_classes))
#
#     # Filter to classes that are in half of mask_cls (old or new classes)
#     half_classes = (
#         unique_classes[: int(len(unique_classes) * 0.3)   ]
#         if mask_cls[0]
#         else unique_classes[int(len(unique_classes) * 0.3): ]
#     )
#     # print(half_classes)
#     # half_classes =unique_classes
#
#
#     # print(half_classes)
#     # Create a mask for labeled samples that belong to the selected half-classes
#     class_mask = np.isin(labeled_targets, half_classes)
#
#     # Get indices of labeled samples belonging to half-classes
#     labeled_half_class_indices = np.where(class_mask)[0]
#
#     # Sample 75% of these indices
#     sample_size = int(class_mask.sum() * sampling_ratio)
#     sampled_indices = np.random.choice(
#         labeled_half_class_indices, size=sample_size, replace=False
#     )
#
#     left_sample_indices = np.setdiff1d(labeled_half_class_indices, sampled_indices, assume_unique = True)
#
#
#     # Create a final mask for all samples, default to False
#     final_mask = np.zeros_like(mask_lab, dtype=bool)
#     final_mask[mask_lab] = np.isin(np.arange(sum(mask_lab)), sampled_indices)
#
#     valid_mask = mask_lab.astype(int) - final_mask.astype(int)
#     # print(mask_lab, final_mask, valid_mask)
#     valid_mask = valid_mask.astype(bool)
#
#     valid_mask_seen_mask = np.zeros_like(mask_lab, dtype=bool)
#     valid_mask_seen_mask[mask_lab] = np.isin(np.arange(sum(mask_lab)), left_sample_indices)
#
#     valid_mask_unseen_mask = valid_mask.astype(int) - valid_mask_seen_mask.astype(int)
#
#
#     return final_mask, valid_mask, valid_mask_seen_mask, valid_mask_unseen_mask



# setting 2
# def sample_mask_lab(mask_lab, mask_cls, targets, sampling_ratio=0.9):
#     # Ensure mask_lab and mask_cls are boolean arrays
#     mask_lab = mask_lab.astype(bool)
#     mask_cls = mask_cls.astype(bool)
#
#     # Filter targets to only include those from labeled points
#     labeled_targets = targets[mask_lab]
#
#     # Get unique classes from labeled points
#     unique_classes = np.unique(labeled_targets)
#     # print(unique_classes)
#     np.random.shuffle(unique_classes)
#     # print(len(unique_classes))
#
#     # Filter to classes that are in half of mask_cls (old or new classes)
#     half_classes = (
#         unique_classes[: int(len(unique_classes) * 0.3)   ]
#         if mask_cls[0]
#         else unique_classes[int(len(unique_classes) * 0.3): ]
#     )
#     # print(half_classes)
#     # half_classes =unique_classes
#
#
#     # print(half_classes)
#     # Create a mask for labeled samples that belong to the selected half-classes
#     class_mask = np.isin(labeled_targets, half_classes)
#
#     # Get indices of labeled samples belonging to half-classes
#     labeled_half_class_indices = np.where(class_mask)[0]
#
#     # Sample 75% of these indices
#     sample_size = int(class_mask.sum() * sampling_ratio)
#     sampled_indices = np.random.choice(
#         labeled_half_class_indices, size=sample_size, replace=False
#     )
#
#     left_sample_indices = np.setdiff1d(labeled_half_class_indices, sampled_indices, assume_unique = True)
#
#
#     # Create a final mask for all samples, default to False
#     final_mask = np.zeros_like(mask_lab, dtype=bool)
#     final_mask[mask_lab] = np.isin(np.arange(sum(mask_lab)), sampled_indices)
#
#     valid_mask = mask_lab.astype(int) - final_mask.astype(int)
#     # print(mask_lab, final_mask, valid_mask)
#     valid_mask = valid_mask.astype(bool)
#
#     valid_mask_seen_mask = np.zeros_like(mask_lab, dtype=bool)
#     valid_mask_seen_mask[mask_lab] = np.isin(np.arange(sum(mask_lab)), left_sample_indices)
#
#     valid_mask_unseen_mask = valid_mask.astype(int) - valid_mask_seen_mask.astype(int)
#
#
#     return final_mask, valid_mask, valid_mask_seen_mask, valid_mask_unseen_mask





# setting 1
# def sample_mask_lab(mask_lab, mask_cls, targets, sampling_ratio=0.75):
#     # Ensure mask_lab and mask_cls are boolean arrays
#     mask_lab = mask_lab.astype(bool)
#     mask_cls = mask_cls.astype(bool)
#
#     # Filter targets to only include those from labeled points
#     labeled_targets = targets[mask_lab]
#
#     # Get unique classes from labeled points
#     unique_classes = np.unique(labeled_targets)
#     # print(unique_classes)
#     np.random.shuffle(unique_classes)
#     # print(len(unique_classes))
#
#     # Filter to classes that are in half of mask_cls (old or new classes)
#     half_classes = (
#         unique_classes[: int(len(unique_classes) * 0.5)   ]
#         if mask_cls[0]
#         else unique_classes[int(len(unique_classes) * 0.5): ]
#     )
#     # print(half_classes)
#     # half_classes =unique_classes
#
#
#     # print(half_classes)
#     # Create a mask for labeled samples that belong to the selected half-classes
#     class_mask = np.isin(labeled_targets, half_classes)
#
#     # Get indices of labeled samples belonging to half-classes
#     labeled_half_class_indices = np.where(class_mask)[0]
#
#     # Sample 75% of these indices
#     sample_size = int(class_mask.sum() * sampling_ratio)
#     sampled_indices = np.random.choice(
#         labeled_half_class_indices, size=sample_size, replace=False
#     )
#
#     left_sample_indices = np.setdiff1d(labeled_half_class_indices, sampled_indices, assume_unique = True)
#
#
#     # Create a final mask for all samples, default to False
#     final_mask = np.zeros_like(mask_lab, dtype=bool)
#     final_mask[mask_lab] = np.isin(np.arange(sum(mask_lab)), sampled_indices)
#
#     valid_mask = mask_lab.astype(int) - final_mask.astype(int)
#     # print(mask_lab, final_mask, valid_mask)
#     valid_mask = valid_mask.astype(bool)
#
#     valid_mask_seen_mask = np.zeros_like(mask_lab, dtype=bool)
#     valid_mask_seen_mask[mask_lab] = np.isin(np.arange(sum(mask_lab)), left_sample_indices)
#
#     valid_mask_unseen_mask = valid_mask.astype(int) - valid_mask_seen_mask.astype(int)
#
#
#     return final_mask, valid_mask, valid_mask_seen_mask, valid_mask_unseen_mask


def get_valid_mask_and_filtered_seen_mask(set_path):
    with open(os.path.join(set_path, "targets.npy"), "rb") as f_targets:
        targets = np.load(f_targets)
    with open(os.path.join(set_path, "mask_cls.npy"), "rb") as f_mask_cls:
        # True if the point is from an old class,
        # False if the point is from a novel class
        mask_cls = np.load(f_mask_cls)
    with open(os.path.join(set_path, "mask_lab.npy"), "rb") as f_mask_lab:
        # True for a labeled point, False for an unlabeled point
        mask_lab = np.load(f_mask_lab)


    filtered_seen_mask, valid_mask, valid_mask_seen_mask, valid_mask_unseen_mask, labeled_targets = sample_mask_lab(
        mask_lab, mask_cls, targets,
    )
    print(f"mask_lab sum: {mask_lab.sum()}")
    print(f"filtered_seen_mask sum: {filtered_seen_mask.sum()}")
    print(f"valid_mask sum: {valid_mask.sum()}, ")

    # print(f"mask_lab len: {len(mask_lab)}")
    # print(f"filtered_seen_mask len: {len(filtered_seen_mask)}")
    # print(f"valid_mask len: {len(valid_mask)}")

    # print(f"valid_mask_seen_mask len: {len(valid_mask_seen_mask)}")
    print(f"valid_mask_seen_mask sum: {valid_mask_seen_mask.sum()}")
    # print(f"valid_mask_unseen_mask len: {len(valid_mask_unseen_mask)}")
    print(f"valid_mask_unseen_mask sum: {valid_mask_unseen_mask.sum()}")
    return filtered_seen_mask, valid_mask, valid_mask_seen_mask, valid_mask_unseen_mask, labeled_targets
