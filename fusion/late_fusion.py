import os
import numpy as np
import argparse
import pickle as pkl


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_run_image_dir", type=str, required=False)
    parser.add_argument("--n_run_audio_dir", type=str, required=False)

    args = parser.parse_args()

    return args


def main():
    args = get_args()

    with open(os.path.join(args.n_run_audio_dir, 'proba.pkl'), "rb") as f:
        audio_proba = pkl.load(f)

    with open(os.path.join(args.n_run_image_dir, 'proba.pkl'), "rb") as f:
        image_proba = pkl.load(f)

    # check if both sets are the same
    assert set(audio_proba.keys()) == set(image_proba.keys())
    devices_names = set(audio_proba.keys())

    audio_acc, image_acc = [], []
    mult_audio_image_acc = []
    add_audio_image_acc = []

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

        y_gnd = int(le.transform([dev_id]))
        # Unimodal
        y_pred_audio = np.argmax(audio_proba[devices_name])
        y_pred_image = np.argmax(image_proba[devices_name])

        audio_acc.append(y_pred_audio == y_gnd)
        image_acc.append(y_pred_image == y_gnd)

        # multiplicative
        mult_audio_image_proba = [a * i for a, i in zip(audio_proba[devices_name], image_proba[devices_name])]
        y_pred_mult_audio_image = np.argmax(mult_audio_image_proba)
        mult_audio_image_acc.append(y_pred_mult_audio_image == y_gnd)

        # # additive
        add_audio_image_proba = [a + i for a, i in zip(audio_proba[devices_name], image_proba[devices_name])]
        y_pred_add_audio_image = np.argmax(add_audio_image_proba)
        add_audio_image_acc.append(y_pred_add_audio_image == y_gnd)

    print(f"Audio acc: {sum(audio_acc) / len(devices_names)}")
    print(f"Image acc: {sum(image_acc) / len(devices_names)}")

    print(f"Multiplicative fusion acc: {sum(mult_audio_image_acc) / len(devices_names)}")
    print(f"Additive fusion acc: {sum(add_audio_image_acc) / len(devices_names)}")

    print()


if __name__ == '__main__':
    """
--n_run_audio_dir
/media/blue/tsingalis/DeviceIdentification/audio/results/Native/fold0/run1/
--n_run_image_dir
/media/blue/tsingalis/DeviceIdentification/image/results/Native/fold0/run1/
    """
    args = get_args()

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
