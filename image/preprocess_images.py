import os
import argparse
import warnings
import numpy as np
from glob import glob
import pickle as pkl
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')

image_types = {"Flat": "flat", "Native": "nat", "NativeFBH": "natFBH", "NativeFBL": "natFBL", "NativeWA": "natWA"}


def extract_features(devices_names, n_classes):

    images_list = []
    targets = []
    for devices_name in devices_names:
        dev_id = devices_name[:3]
        if not any([dev_id in cn for cn in class_names]):
            continue
        if dev_id == 'D01' or dev_id == 'D26':
            dev_id = 'D01-D26'
        if dev_id == 'D06' or dev_id == 'D15':
            dev_id = 'D06-D15'
        elif dev_id == 'D02' or dev_id == 'D10':
            dev_id = 'D02-D10'
        elif dev_id == 'D05' or dev_id == 'D14' or dev_id == 'D18':
            dev_id = 'D05-D14-D18'
        elif dev_id == 'D29' or dev_id == 'D34':
            dev_id = 'D29-D34'

        target = int(le.transform([dev_id]))
        targets.append(target)

        n_frames = len(glob(os.path.join(args.vision_frames_dir, devices_name, '*.jpg')))
        # n_frames = round(n_images * 0.8)
        images_pbar = tqdm(range(1, n_frames+1, 2), position=0, leave=True, total=n_frames)
        for frames_index in images_pbar:
            img_path = os.path.join(args.vision_frames_dir, devices_name, 'frame{:06d}.jpg'.format(frames_index))
            # image = Image.open(img_path)
            images_pbar.set_description("Processing: Device name %s -- frame: %s" % (devices_name, Path(img_path).stem))
            images_list.append({"image_name": Path(img_path).stem,
                                "device_name": devices_name,
                                "target": target,
                                "n_classes": n_classes})
    print(f'Unique labels {len(np.unique(targets))}')

    return images_list


def main():
    tr_df = pd.read_csv(f'{args.split_dir}/{args.visual_content}/train_fold{args.n_fold}.csv')
    tst_df = pd.read_csv(f'{args.split_dir}/{args.visual_content}/test_fold{args.n_fold}.csv')
    val_df = pd.read_csv(f'{args.split_dir}/{args.visual_content}/val_fold{args.n_fold}.csv')

    n_classes = len(np.unique(tr_df['dev_numerical_id'].values))

    training_values = extract_features(tr_df.dev_alphabetical_id.values, n_classes)
    with open(f"{args.output_dir}/train_128images_{args.visual_content}_fold{args.n_fold}.pkl", "wb") as handler:
        pkl.dump(training_values, handler, protocol=pkl.HIGHEST_PROTOCOL)

    validation_values = extract_features(val_df.dev_alphabetical_id.values, n_classes)
    with open(f"{args.output_dir}/valid_128images_{args.visual_content}_fold{args.n_fold}.pkl",
              "wb") as handler:
        pkl.dump(validation_values, handler, protocol=pkl.HIGHEST_PROTOCOL)

    test_values = extract_features(tst_df.dev_alphabetical_id.values, n_classes)
    with open(f"{args.output_dir}/test_128images_{args.visual_content}_fold{args.n_fold}.pkl", "wb") as handler:
        pkl.dump(test_values, handler, protocol=pkl.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    """
    --output_dir /media/blue/tsingalis/IARIADevIDFusion/image/ 
    --vision_frames_dir /media/red/sharedFolder/Datasets/VISION/keyframeExtraction/extractedFrames/
    --visual_content YT 
    --n_fold 0 
    --split_dir /media/blue/tsingalis/IARIADevIDFusion/splits/JoI_splits
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision_frames_dir", type=str, required=True)
    parser.add_argument("--split_dir", type=str, required=True)
    parser.add_argument("--n_fold", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_mels", default=128, type=int)
    parser.add_argument('--visual_content',
                        choices=['YT', 'Native', 'WA'], required=True)

    args = parser.parse_args()

    class_names = ['D01-D26', 'D02-D10', 'D03', 'D04',
                   'D05-D14-D18', 'D06-D15',
                   'D07', 'D08', 'D09', 'D11', 'D12',
                   'D13', 'D16', 'D17', 'D19', 'D20',
                   'D21', 'D22', 'D23', 'D24', 'D25', 'D27', 'D28',
                   'D29-D34', 'D30', 'D31', 'D32', 'D33', 'D35']

    class_names.remove('D04')
    class_names.remove('D12')
    class_names.remove('D17')
    class_names.remove('D22')

    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder().fit(class_names)

    # show how many classes there are
    print(list(le.classes_))

    main()
