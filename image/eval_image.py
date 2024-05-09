import os
import json
import torch
import argparse
import numpy as np
import pickle as pkl
from tqdm.auto import tqdm
from utils import save_cm_fig
from collections import Counter
from models.audio_image_models import *
from image.dataset import ImageDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=False)
    parser.add_argument("--project_dir", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--vision_frames_dir", type=str, required=True)

    # parameter #
    parser.add_argument("--cuda", default=0, type=int, help='Specify cuda number')
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--train_batch", default=8, type=int)
    parser.add_argument("--valid_batch", default=8, type=int)
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-3, type=int)

    parser.add_argument("--split_dir", type=str, required=True)

    # model #
    parser.add_argument("--model", default="densenet", type=str,
                        choices=["DenseNet201", "ResNet50", "InceptionV3",
                                 "ResNet18", "MobileNetV3Small", "MobileNetV3Large", "SqueezeNet1_1"])

    parser.add_argument('--visual_content',
                        choices=['YT', 'WA', 'Native'], required=False)

    parser.add_argument("--n_fold", type=int, required=False)

    parser.add_argument("--n_run", type=int, required=False)

    args = parser.parse_args()

    with open(os.path.join(args.results_dir,
                           args.visual_content,
                           f"fold{args.n_fold}",
                           f'run{args.n_run}', 'args.json'), 'rt') as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)

    return args


def eval(model, device, test_loader, n_classes, video_ids_dict):
    train_loss, total_acc, total_cnt = 0, 0, 0
    model.eval()

    n_batches = 0
    with torch.no_grad():
        for data in tqdm(test_loader, desc='Predicting...'):
            inputs = data[0].to(device)
            target_labels = data[1].squeeze(1).to(device)
            device_names = data[2]

            outputs = model(inputs)
            pred_proba = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()

            _, pred_labels = torch.max(outputs.data, 1)
            total_acc += torch.sum((pred_labels == target_labels).float()).item()
            total_cnt += target_labels.size(0)

            target_labels = target_labels.cpu()
            pred_labels = pred_labels.cpu()

            for device_name, pred_label, target_label in zip(device_names, pred_labels, target_labels):
                video_ids_dict[device_name]["gnd_labels"].append(target_label.item())
                video_ids_dict[device_name]["pred_labels"].append(pred_label.item())
                video_ids_dict[device_name]["proba"].append(pred_proba[device_names.index(device_name)].tolist())

            n_batches += 1

    per_dev_class_results, per_dev_proba = {}, {}
    target_labels_per_dev, pred_labels_per_dev = [], []
    for key, values in video_ids_dict.items():
        bins, counts = zip(*Counter(values["pred_labels"]).items())
        major_label = bins[np.argmax(counts)]
        gnd_label = np.unique(values["gnd_labels"])
        target_labels_per_dev.append(gnd_label)
        pred_labels_per_dev.append(major_label)
        per_dev_class_results.update({key: {"major_label": int(major_label), "gnd_label": int(gnd_label)}})
        per_dev_proba.update({key: np.mean(values["proba"], axis=0).tolist()})

    return per_dev_proba


def main():
    print('Loading model...')
    if args.model == "DenseNet201":
        model = DenseNet201(num_classes=num_classes).to(device)
    elif args.model == "ResNet50":
        model = ResNet50(num_classes=num_classes).to(device)
    elif args.model == "InceptionV3":
        model = InceptionV3(weights=models.Inception_V3_Weights.DEFAULT, num_classes=num_classes).to(device)
    elif args.model == "ResNet18":
        model = ResNet18(weights=models.ResNet18_Weights.DEFAULT, num_classes=num_classes).to(device)
    elif args.model == "MobileNetV3Small":
        model = MobileNetV3Small(weights=models.MobileNet_V3_Small_Weights.DEFAULT,
                                 num_classes=num_classes).to(device)
    elif args.model == "MobileNetV3Large":
        model = MobileNetV3Large(weights=models.MobileNet_V3_Large_Weights.DEFAULT,
                                 num_classes=num_classes).to(device)
    elif args.model == "SqueezeNet1_1":
        model = SqueezeNet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT,
                              num_classes=num_classes).to(device)

    checkpoint = torch.load(os.path.join(str(args.results_dir),
                                         args.visual_content,
                                         f"fold{args.n_fold}",
                                         f"run{args.n_run}", "model_best.ckpt"),
                            map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])  # already is ['state_dict']
    model = model.to(device)
    model.eval()

    # show how many classes there are
    print(list(le.classes_))

    # Predicted result
    per_dev_proba = eval(model, device, test_loader, n_classes=len(class_names), video_ids_dict=video_ids_dict)

    with open(os.path.join(args.results_dir, args.visual_content,
                           f"fold{args.n_fold}", f"run{args.n_run}", 'proba.pkl'), "wb") as handler:
        pkl.dump(per_dev_proba, handler, protocol=pkl.HIGHEST_PROTOCOL)

    with open(os.path.join(args.results_dir, args.visual_content,
                           f"fold{args.n_fold}", f"run{args.n_run}", 'proba.pkl'), "rb") as f:
        proba = pkl.load(f)


def map_dev_id_to_classes(dev_id):
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

    return dev_id


if __name__ == '__main__':

    """
--n_run
1
--visual_content
YT
--n_fold
0
--vision_frames_dir
/media/red/sharedFolder/Datasets/VISION/keyframeExtraction/extractedFrames
--project_dir
/media/blue/tsingalis/IARIADevIDFusion/image/
--results_dir
/media/blue/tsingalis/IARIADevIDFusion/image/results/
--split_dir
/media/blue/tsingalis/IARIADevIDFusion/splits/JoI_splits
    """
    args = get_args()
    cuda_num = str(args.cuda)
    device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    class_names = ['D01-D26', 'D02-D10', 'D03', 'D04', 'D05-D14-D18',
                   'D06-D15', 'D07', 'D08', 'D09', 'D11', 'D12',
                   'D13', 'D16', 'D17', 'D19', 'D20', 'D21',
                   'D22', 'D23', 'D24', 'D25', 'D27', 'D28',
                   'D29-D34', 'D30', 'D31', 'D32', 'D33', 'D35']

    class_names.remove('D04')
    class_names.remove('D12')
    class_names.remove('D17')
    class_names.remove('D22')

    num_classes = len(class_names)

    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder().fit(class_names)

    pkl_dir_tr = f"{args.project_dir}/preprocessed_images/test_128images_{args.visual_content}_fold{args.n_fold}.pkl"
    test_set = ImageDataset(pkl_dir_tr, args.vision_frames_dir)
    print(f"Number of Test samples {len(test_set)}")

    test_loader = DataLoader(test_set, batch_size=args.valid_batch, shuffle=True,
                             pin_memory=True, num_workers=args.num_workers)

    video_ids_dict = {}

    for devices_name in set([t['device_name'] for t in test_set.data]):

        dev_id = devices_name[:3]
        if not any([dev_id in cn for cn in class_names]):
            continue

        video_ids_dict.update({devices_name: {
            "gnd_labels": list(),
            "pred_labels": list(),
            "proba": list()
        }
        })

    main()
