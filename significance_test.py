import os
import numpy as np
import argparse
import pickle as pkl

from mlxtend.evaluate import mcnemar_table
from mlxtend.evaluate import ftest


def get_args():
    parser = argparse.ArgumentParser()

    # parser.add_argument("--n_run_image_dir", type=str, required=False)
    # parser.add_argument("--n_run_audio_dir", type=str, required=False)

    parser.add_argument("--n_run_audio", type=str, required=True)
    parser.add_argument("--n_run_image", type=str, required=True)

    parser.add_argument('--visual_content',
                        choices=['YT', 'WA', 'Native'], required=True)

    parser.add_argument("--audio_project_dir",
                        default="/media/blue/tsingalis/DeviceIdentification/audio/",
                        type=str, required=True)
    parser.add_argument("--image_project_dir",
                        default="/media/blue/tsingalis/DeviceIdentification/image/",
                        type=str, required=True)

    parser.add_argument("--n_fold", type=int, required=True)

    args = parser.parse_args()

    return args


def significance_test(y_target, y_model1, y_model2, task_label):
    tb = mcnemar_table(y_target=y_target,
                       y_model1=y_model1,
                       y_model2=y_model2)
    # print(tb)
    # _, p = mcnemar(ary=tb, corrected=True, exact=False)

    _, p = ftest(y_target, y_model1, y_model2)
    # _, p = cochrans_q(y_target, y_model1, y_model2)

    # print('chi-squared:', chi2)
    # print('p-value:', p)
    """
    If the p-value is lower than our chosen significance level (p=0.05), we can reject the null 
    hypothesis that the two model's performances are equal.
    References:
    [1] https://stats.stackexchange.com/questions/26271/comparing-two-classifier-accuracy-results-for-statistical-significance-with-t-te
    [2] https://machinelearningmastery.com/statistical-significance-tests-for-comparing-machine-learning-algorithms/
    [3] https://sebastianraschka.com/blog/2018/model-evaluation-selection-part4.html
    """
    if p < 0.05:
        print(f"{task_label} -- p-value is: {p} -- Thus, we can reject the null "
              f"hypothesis that the two model's performances are equal.")
    else:
        print(f"{task_label} -- p-value is: {p} -- Thus, we can not reject the null "
              f"hypothesis that the two model's performances are equal.")

    return p


def main():
    args = get_args()

    n_run_audio_dir = os.path.join(args.audio_project_dir, 'results',
                                   args.visual_content,
                                   f'fold{args.n_fold}',
                                   f'run{args.n_run_audio}')
    with open(os.path.join(str(n_run_audio_dir), 'proba.pkl'), "rb") as f:
        audio_proba = pkl.load(f)

    n_run_image_dir = os.path.join(args.image_project_dir, 'results',
                                   args.visual_content,
                                   f'fold{args.n_fold}',
                                   f'run{args.n_run_image}')
    with open(os.path.join(str(n_run_image_dir), 'proba.pkl'), "rb") as f:
        image_proba = pkl.load(f)

    # check if both sets are the same
    assert set(audio_proba.keys()) == set(image_proba.keys())
    devices_names = set(audio_proba.keys())

    audio_acc, image_acc = [], []
    mult_audio_image_acc = []
    add_audio_image_acc = []

    # The correct target (class) labels
    y_target, y_product_rule, y_sum_rule, y_audio, y_video = [], [], [], [], []

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

        y_target.append(y_gnd)

        y_audio.append(y_pred_audio)
        y_video.append(y_pred_image)

        y_product_rule.append(y_pred_mult_audio_image)
        y_sum_rule.append(y_pred_add_audio_image)

    y_target, y_product_rule, y_sum_rule = np.array(y_target), np.array(y_product_rule), np.array(y_sum_rule)
    y_audio, y_video = np.array(y_audio), np.array(y_video)

    significance_test(y_target, y_product_rule, y_sum_rule, 'Product-Sum Significance test')
    significance_test(y_target, y_product_rule, y_audio, 'Product-Audio Significance test')
    significance_test(y_target, y_product_rule, y_video, 'Product-Video Significance test')

    significance_test(y_target, y_sum_rule, y_audio, 'Sum-Audio Significance test')
    significance_test(y_target, y_sum_rule, y_video, 'Sum-Video Significance test')

    """
    Accuracy results
    """

    print(f"Audio acc: {sum(audio_acc) / len(devices_names)}")
    print(f"Image acc: {sum(image_acc) / len(devices_names)}")

    print(f"Multiplicative fusion acc: {sum(mult_audio_image_acc) / len(devices_names)}")
    print(f"Additive fusion acc: {sum(add_audio_image_acc) / len(devices_names)}")

    print()


if __name__ == '__main__':
    """
--n_run_audio
1
--audio_project_dir
/media/blue/tsingalis/IARIADevIDFusion/audio
--n_run_image
1
--image_project_dir
/media/blue/tsingalis/IARIADevIDFusion/image
--n_fold
0
--visual_content
YT
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
