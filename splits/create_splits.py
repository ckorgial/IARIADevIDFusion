import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

n_splits = 5
# kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)

from sklearn.model_selection import StratifiedKFold

kf = StratifiedKFold(n_splits=n_splits)


def main():
    save_path = os.path.join(args.output_dir, args.visual_content)
    Path(save_path).mkdir(parents=True, exist_ok=True)

    dev_names = [filename for filename in os.listdir(args.input_dir) if
                 os.path.isdir(os.path.join(args.input_dir, filename))]

    # class_names = ['D01', 'D02', 'D03', 'D04', 'D05', 'D06', 'D07', 'D08', 'D09', 'D10', 'D11', 'D12', 'D13', 'D14',
    #                'D15', 'D16', 'D17', 'D18', 'D19', 'D20', 'D21', 'D22', 'D23', 'D24', 'D25', 'D26', 'D27', 'D28',
    #                'D29', 'D30', 'D31', 'D32', 'D33', 'D34', 'D35']

    class_names = ['D01-D26', 'D02-D10', 'D03', 'D04', 'D05-D14-D18', 'D06-D15', 'D07', 'D08', 'D09', 'D11', 'D12',
                   'D13', 'D16', 'D17', 'D19', 'D20', 'D21', 'D22', 'D23', 'D24', 'D25', 'D27', 'D28',
                   'D29-D34', 'D30', 'D31', 'D32', 'D33', 'D35']

    class_names.remove('D04')
    class_names.remove('D12')
    class_names.remove('D17')
    class_names.remove('D22')

    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder().fit(class_names)

    # show how many classes there are
    print(list(le.classes_))

    dev_numerical_id_list = []
    dev_alphabetical_id_list = []
    for dev_name in dev_names:
        if 'flat' in dev_name:
            continue
        if (('YT' in dev_name and 'YT' in args.visual_content)
                or ('WA' in dev_name and 'WA' in args.visual_content)
                or ('YT' not in dev_name and 'WA' not in dev_name and 'Native' in args.visual_content)):
            str_dev_id = dev_name[:3]
            if not any([str_dev_id in cn for cn in class_names]):
                continue
            if str_dev_id == 'D01' or str_dev_id == 'D26':
                str_dev_id = 'D01-D26'
            if str_dev_id == 'D06' or str_dev_id == 'D15':
                str_dev_id = 'D06-D15'
            elif str_dev_id == 'D02' or str_dev_id == 'D10':
                str_dev_id = 'D02-D10'
            elif str_dev_id == 'D05' or str_dev_id == 'D14' or str_dev_id == 'D18':
                str_dev_id = 'D05-D14-D18'
            elif str_dev_id == 'D29' or str_dev_id == 'D34':
                str_dev_id = 'D29-D34'

            dev_id = int(le.transform([str_dev_id]))
            dev_numerical_id_list.append(dev_id)
            dev_alphabetical_id_list.append(dev_name)

    print(f'Number of labels: {len(np.unique(dev_numerical_id_list))}')

    kf_pairs = list(kf.split(dev_alphabetical_id_list, dev_numerical_id_list))
    for i, (train_val_index, test_index) in enumerate(kf_pairs):
        train_index, val_index = train_test_split(train_val_index, train_size=0.9)
        tr_data_pair = [(dev_alphabetical_id_list[j], dev_numerical_id_list[j]) for j in train_index]
        val_data_pair = [(dev_alphabetical_id_list[j], dev_numerical_id_list[j]) for j in val_index]
        tst_data_pair = [(dev_alphabetical_id_list[j], dev_numerical_id_list[j]) for j in test_index]

        save = True
        if save:
            df = pd.DataFrame(tr_data_pair, columns=['dev_alphabetical_id', 'dev_numerical_id'])
            df.to_csv(f'{save_path}/train_fold{i}.csv', index=False)

            df = pd.DataFrame(tst_data_pair, columns=['dev_alphabetical_id', 'dev_numerical_id'])
            df.to_csv(f'{save_path}/test_fold{i}.csv', index=False)

            df = pd.DataFrame(val_data_pair, columns=['dev_alphabetical_id', 'dev_numerical_id'])
            df.to_csv(f'{save_path}/val_fold{i}.csv', index=False)

    all_train, all_test, all_valid = [], [], []

    for cnt_fold in range(n_splits):
        all_train.extend(pd.read_csv(f'{save_path}/train_fold{cnt_fold}.csv')['dev_alphabetical_id'])
        all_test.extend(pd.read_csv(f'{save_path}/test_fold{cnt_fold}.csv')['dev_alphabetical_id'])
        all_valid.extend(pd.read_csv(f'{save_path}/val_fold{cnt_fold}.csv')['dev_alphabetical_id'])

    assert set(all_test) == set(dev_alphabetical_id_list)
    print()


if __name__ == '__main__':
    """
    --output_dir /media/blue/tsingalis/IARIADevIDFusion/splits/JoI_splits 
    --input_dir /media/blue/tsingalis/IARIADevIDFusion/audio/extractedWav/
    --visual_content Native
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sr", default=22050, type=int)
    parser.add_argument("--n_splits", default=4, type=int)
    parser.add_argument('--visual_content',
                        choices=['YT', 'WA', 'Native'], required=True)

    args = parser.parse_args()

    main()
