import ast
from tqdm import tqdm
from glob import glob
from pathlib import Path
from multiprocessing import cpu_count, Pool
from sklearn.model_selection import train_test_split

import os
import cv2
import subprocess


def get_frame_types(video_fn):
    """
    Reference:
    https://stackoverflow.com/questions/42798634/extracting-keyframes-python-opencv
    """
    command = 'ffprobe -v error -show_entries frame=pict_type -of default=noprint_wrappers=1'.split()
    out = subprocess.check_output(command + [video_fn]).decode()
    frame_types = out.replace('pict_type=', '').split()
    return zip(range(len(frame_types)), frame_types)


def get_keyframes(video_fn, key_frame_type='I'):
    """
    Reference:
    https://stackoverflow.com/questions/42798634/extracting-keyframes-python-opencv
    """
    frame_types = get_frame_types(video_fn)
    i_frames = [x[0] for x in frame_types if x[1] == key_frame_type]
    if i_frames:
        basename = os.path.splitext(os.path.basename(video_fn))[0]
        cap = cv2.VideoCapture(video_fn)
        for frame_no in i_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            out_name = basename + '_i_frame_' + str(frame_no) + '.jpg'
            cv2.imwrite(out_name, frame)
            print('Saved: ' + out_name)
        cap.release()
    else:
        print('No I-frames in ' + video_fn)


def process_brand(dev_ids):
    for n_dev, dev_id in enumerate(dev_ids):

        natural_videos_flat = {}
        for type in ['*.mp4', '*.mov', '*.3gp']:
            flat_path = os.path.join(devices_folder, dev_id, 'videos', 'flat', type)
            if os.path.exists(Path(flat_path).parent):
                for i, x in enumerate(glob(flat_path)):
                    natural_videos_flat.update({f'flat{i}': x})

        natural_videos_flatWA = {}
        for type in ['*.mp4', '*.mov', '*.3gp']:
            flatWA_path = os.path.join(devices_folder, dev_id, 'videos', 'flatWA', type)
            if os.path.exists(Path(flatWA_path).parent):
                for i, x in enumerate(glob(flatWA_path)):
                    natural_videos_flatWA.update({f'flatWA{i}': x})

        natural_videos_indoor = {}
        for type in ['*.mp4', '*.mov', '*.3gp']:
            indoor_path = os.path.join(devices_folder, dev_id, 'videos', 'indoor', type)
            if os.path.exists(Path(indoor_path).parent):
                for i, x in enumerate(glob(indoor_path)):
                    natural_videos_indoor.update({f'indoor{i}': x})

        natural_videos_outdoor = {}
        for type in ['*.mp4', '*.mov', '*.3gp']:
            outdoor_path = os.path.join(devices_folder, dev_id, 'videos', 'outdoor', type)
            if os.path.exists(Path(outdoor_path).parent):
                for i, x in enumerate(glob(outdoor_path)):
                    natural_videos_outdoor.update({f'outdoor{i}': x})

        natural_videos_indoorWA = {}
        for type in ['*.mp4', '*.mov', '*.3gp']:
            indoorWA_path = os.path.join(devices_folder, dev_id, 'videos', 'indoorWA', type)
            if os.path.exists(Path(indoorWA_path).parent):
                for i, x in enumerate(glob(indoorWA_path)):
                    natural_videos_indoorWA.update({f'indoorWA{i}': x})

        natural_videos_indoorYT = {}
        for type in ['*.mp4', '*.mov', '*.3gp']:
            indoorYT_path = os.path.join(devices_folder, dev_id, 'videos', 'indoorYT', type)
            if os.path.exists(Path(indoorYT_path).parent):
                for i, x in enumerate(glob(indoorYT_path)):
                    natural_videos_indoorYT.update({f'indoorYT{i}': x})

        natural_videos_outdoorWA = {}
        for type in ['*.mp4', '*.mov', '*.3gp']:
            outdoorWA_path = os.path.join(devices_folder, dev_id, 'videos', 'outdoorWA', type)
            if os.path.exists(Path(outdoorWA_path).parent):
                for i, x in enumerate(glob(outdoorWA_path)):
                    natural_videos_outdoorWA.update({f'outdoorWA{i}': x})

        natural_videos_outdoorYT = {}
        for type in ['*.mp4', '*.mov', '*.3gp']:
            outdoorYT_path = os.path.join(devices_folder, dev_id, 'videos', 'outdoorYT', type)
            if os.path.exists(Path(outdoorYT_path).parent):
                for i, x in enumerate(glob(outdoorYT_path)):
                    natural_videos_outdoorYT.update({f'outdoorYT{i}': x})

        natural_videos = {**natural_videos_flat,
                          **natural_videos_flatWA,
                          **natural_videos_indoor,
                          **natural_videos_outdoor,
                          **natural_videos_indoorWA,
                          **natural_videos_indoorYT,
                          **natural_videos_outdoorWA,
                          **natural_videos_outdoorYT}

        excluded_devs = ['D04', 'D12', 'D17', 'D22']
        for indoor_or_outdoor, natural_video in natural_videos.items():
            dev_id = Path(natural_video).stem[:3]
            if dev_id in excluded_devs or 'flat' in indoor_or_outdoor:
                continue

            save_folder = os.path.join(output_folder, Path(natural_video).stem)
            # if os.path.exists(save_folder):
            #     continue
            Path(save_folder).mkdir(parents=True, exist_ok=True)

            cap = cv2.VideoCapture(natural_video)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # total_frames = 400
            for frame_no in tqdm(range(0, total_frames),
                                 desc=f'Device {dev_id} ({n_dev}/{len(dev_ids)}) '
                                      f'-- Video: {Path(natural_video).stem} -- Process frames...',
                                 position=0, leave=True):
                if os.path.isfile(os.path.join(save_folder, 'frame{:06d}.jpg'.format(frame_no + 1))):
                    continue

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
                ret, frame = cap.read()
                if ret:
                    cv2.imwrite(os.path.join(save_folder, 'frame{:06d}.jpg'.format(frame_no + 1)), frame)

            cap.release()


def main():
    dev_ids = [name for name in os.listdir(devices_folder)]

    process_brand(dev_ids=dev_ids)

    print('That is all folks...')


if __name__ == '__main__':
    root_dir_vision = '/media/red/sharedFolder/Datasets'
    devices_folder = f'{root_dir_vision}/VISION/dataset/'
    output_folder = '/media/blue/tsingalis/IARIADevIDFusion/datasets/VISION/extractedFrames/'

    main()
