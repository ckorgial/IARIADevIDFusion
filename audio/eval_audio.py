import os
import json
import torch
import argparse
import numpy as np
import pickle as pkl
from utils import save_cm_fig
from dataset import AudioDataset
from torch.utils.data import DataLoader
from models.audio_image_models import *
from sklearn.metrics import confusion_matrix


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_dir", type=str, required=False)
    parser.add_argument("--results_dir", type=str, required=False)

    # parameter #
    parser.add_argument("--valid_batch", default=8, type=int)
    parser.add_argument("--num_workers", default=8, type=int)

    parser.add_argument("--mel_dir", type=str, required=True)

    # model #
    parser.add_argument("--model", default="densenet", type=str,
                        choices=["DenseNet201", "ResNet50", "InceptionV3",
                                 "ResNet18", "MobileNetV3Small", "MobileNetV3Large", "SqueezeNet1_1"])

    parser.add_argument('--visual_content',
                        choices=['YT', 'WA', 'Native'],
                        required=True)

    parser.add_argument("--n_fold", type=int, required=True)

    parser.add_argument("--n_run", type=int, required=False)

    args = parser.parse_args()

    with open(os.path.join(args.results_dir, args.visual_content,
                           f"fold{args.n_fold}", f'run{args.n_run}', 'args.json'), 'rt') as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)

    return args


def eval(model, device, test_loader, n_classes):
    train_loss, total_acc, total_cnt = 0, 0, 0
    model.eval()

    per_dev_proba = {}
    with torch.no_grad():
        for data in test_loader:
            inputs = data[0].to(device)
            target = data[1].squeeze(1).to(device)
            device_names = data[2]

            outputs = model(inputs)
            pred_proba = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()

            _, pred_label = torch.max(outputs.data, 1)
            total_acc += torch.sum((pred_label == target).float()).item()
            total_cnt += target.size(0)

            for device_name in device_names:
                per_dev_proba.update({device_name: pred_proba[device_names.index(device_name)].tolist()})

    return per_dev_proba


def main():
    args = get_args()
    cuda_num = str(args.cuda)
    device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    tst_split_dir = os.path.join(args.split_dir, args.visual_content,
                                 f'test_fold{args.n_fold}.csv')

    test_set = AudioDataset(tst_split_dir, args.mel_dir)
    print(f"Number of Test samples {len(test_set)}")

    test_loader = DataLoader(test_set, batch_size=args.valid_batch, shuffle=True,
                             pin_memory=True, num_workers=args.num_workers)

    print('Loading model...')
    num_classes = 25
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

    checkpoint = torch.load(os.path.join(args.results_dir, args.visual_content, f"fold{args.n_fold}",
                                         f"run{args.n_run}", "model_best.ckpt"),
                            map_location='cpu')

    model.load_state_dict(checkpoint['state_dict'])  # already is ['state_dict']
    model = model.to(device)
    model.eval()

    # target_names = ['D01', 'D02', 'D03', 'D04', 'D05', 'D06', 'D07', 'D08', 'D09', 'D10', 'D11', 'D12', 'D13', 'D14',
    #                 'D15', 'D16', 'D17', 'D18', 'D19', 'D20', 'D21', 'D22', 'D23', 'D24', 'D25', 'D26', 'D27', 'D28',
    #                 'D29', 'D30', 'D31', 'D32', 'D33', 'D34', 'D35']

    target_names = ['D01-D26', 'D02-D10', 'D03', 'D04',
                    'D05-D14-D18', 'D06-D15', 'D07', 'D08', 'D09', 'D11', 'D12',
                    'D13', 'D16', 'D17', 'D19', 'D20', 'D21', 'D22', 'D23', 'D24', 'D25', 'D27', 'D28',
                    'D29-D34', 'D30', 'D31', 'D32', 'D33', 'D35']

    target_names.remove('D04')
    target_names.remove('D12')
    target_names.remove('D17')
    target_names.remove('D22')

    # Predicted result
    print('Predicting...')
    per_dev_proba = eval(model, device, test_loader, n_classes=len(target_names))

    with open(os.path.join(args.results_dir, args.visual_content, f"fold{args.n_split}",
                           f"run{args.n_run}", 'proba.pkl'), "wb") as handler:
        pkl.dump(per_dev_proba, handler, protocol=pkl.HIGHEST_PROTOCOL)

    with open(os.path.join(args.results_dir, args.visual_content, f"fold{args.n_split}",
                           f"run{args.n_run}", 'proba.pkl'), "rb") as f:
        proba = pkl.load(f)


if __name__ == '__main__':
    """
--n_run
1
--results_dir
/media/blue/tsingalis/IARIADevIDFusion/audio/results/
--project_dir
/media/blue/tsingalis/IARIADevIDFusion/audio/
--mel_dir
/media/blue/tsingalis/IARIADevIDFusion/audio/extractedMel
--visual_content
YT
--n_fold
0
    """
    main()
