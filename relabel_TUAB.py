""" Functions to move folders in the TUAB dataset, effectively relabelling normal/abnormal, based on data in the
TUAB_relabel_catalog.csv.

** To apply the labels from the csv file **
import relabel_TUAB as rlb
rlb.relabel(your_path_to_main_TUAB_directory)

** To undo the above **
rlb.undo_relabel(your_path_to_main_TUAB_directory)

** To exclude all cases mentioned in the csv file **
rlb.relabel(your_path_to_main_TUAB_directory, exclude_all=True)

** To undo the above **
rlb.undo_relabel(your_path_to_main_TUAB_directory, exclude_all=True)

Created by David Western, July 2021
"""

import pandas as pd
import os
import warnings


def relabel(TUAB_dir, exclude_all=False):
    df = pd.read_csv('TUAB_relabel_catalog.csv')

    for index, row in df.iterrows():
        ff = os.path.join(TUAB_dir, row['Original Path'])
        if not os.path.isfile(ff):
            warnings.warn(f"Failed to locate {ff}. Skipping.")
            continue
        # Dissect file path
        orig_path, fn = os.path.split(ff)
        fid, ext = os.path.splitext(fn)
        head, old_dir = os.path.split(orig_path)

        # Decide where to put it
        if exclude_all or row['New label'] == 'Exclude':
            new_dir = os.path.join(TUAB_dir, 'Exclude')
        elif row['New label'] == 'Abnormal':
            new_dir = orig_path.replace('\\normal', '\\abnormal')
        elif row['New label'] == 'Normal':
            new_dir = orig_path.replace('\\abnormal', '\\normal')
        else:
            raise ValueError(f"Unrecognised label: {row['New label']}")

        # Move all files with the same base name
        directory = os.fsencode(orig_path)
        for file in os.listdir(directory):
            fn = os.fsdecode(file)
            ff = os.path.join(orig_path, fn)
            if fn.startswith(fid):
                new_ff = os.path.join(new_dir, fn)
                os.makedirs(new_dir, exist_ok=True)
                print(f"Moving {ff} to {new_ff}.")
                os.rename(ff, new_ff)


def undo_relabel(TUAB_dir, exclude_all=False):

    df = pd.read_csv('TUAB_relabel_catalog.csv')

    for index, row in df.iterrows():
        # Get current path in which file sits
        orig_ff = os.path.join(TUAB_dir,row['Original Path'])
        orig_path, fn = os.path.split(orig_ff)
        if exclude_all or row['New label'] == 'Exclude':
            current_path = os.path.join(TUAB_dir,'Exclude')
        else:
            # Correct for the relabelling that has presumably been applied previously.
            if row['New label'] == 'Abnormal':
                current_path = orig_path.replace('\\normal','\\abnormal')
            elif row['New label'] == 'Normal':
                current_path = orig_path.replace('\\abnormal','\\normal')
            else:
                raise ValueError(f"Unrecognised label: {row['New label']}")

        # Dissect file path
        fid, ext = os.path.splitext(fn)

        # Move all files with the same base name
        directory = os.fsencode(current_path)
        for file in os.listdir(directory):
            fn = os.fsdecode(file)
            ff = os.path.join(current_path, fn)
            if fn.startswith(fid):
                new_ff = os.path.join(orig_path, fn)
                os.makedirs(orig_path, exist_ok=True)
                print(f"Moving {ff} to {new_ff}.")
                os.rename(ff,new_ff)

    exclude_dir = os.path.join(TUAB_dir,'Exclude')
    if os.path.isdir(exclude_dir):
        os.rmdir(exclude_dir)



undo_relabel('H:\TUAB', exclude_all=True)