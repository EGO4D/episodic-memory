import os
import sys
import json
import argparse
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool

sys.path.append('./Camera_Intrinsics_API/')
from extract_frames import FrameExtractor

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clips_json",
        type=str,
        default='',
        help="a json file with all clip uids to be parsed for each split split",
    )
    parser.add_argument(
        "--split",
        type=str,
        default='',
        help="split set to process. If not specified process all clips.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default='data/clips/',
        help="Input folder with the clips.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='data/clips_frames/',
        help="Output folder with the clips.",
    )
    parser.add_argument(
        "--j",
        type=int,
        default=1,
        help="Number of parallel processes",
    )
    args = parser.parse_args()

    clip_dir = args.input_dir
    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    clips_filenames = os.listdir(clip_dir)
    if args.clips_json:
        clips_json = json.load(open(args.clips_json, 'r'))
        if args.split:
            split = args.split
            clips_filenames = [x for x in clips_filenames if\
                               x.split('.')[0] in clips_json[split]
                              ]
            num_clips = len(clips_filenames)
            print(f'Parsing clips from {split} set - {num_clips} clips total')

        else:
            all_clips_json = clips_json['train']+\
                             clips_json['val']+\
                             clips_json['test']

            clips_filenames = [x for x in clips_filenames if\
                               x.split('.')[0] in all_clips_json
                              ]
            num_clips = len(clips_filenames)
            print(f'Parsing clips from ALL sets - {num_clips} clips total')
    else:
        num_clips = len(clips_filenames)
        print(f'Parsing ALL clips - {num_clips} clips total')

    def frame_extractor(inputs):

        clip_filename = inputs['input']
        clip_directory=inputs['clip_dir']
        output_directory=inputs['output_dir']

        fe = FrameExtractor()

        filename = os.path.join(clip_directory, clip_filename)
        clip_name_uid = clip_filename.split('.')[0]
        output_dir_clip = os.path.join(output_directory, clip_name_uid)

        # skip if already processed
        if os.path.isdir(output_dir_clip):
            if len(os.listdir(output_dir_clip)) > 2000:
                return

        Path(output_dir_clip).mkdir(parents=True, exist_ok=True)

        fe.extract(filename, output_dir_clip)

    inputs = [{'input':x,
               'clip_dir':clip_dir,
               'output_dir':output_dir} for x in clips_filenames]

    pool = Pool(args.j)

    _ = list(
        tqdm(
            pool.imap_unordered(
                frame_extractor, inputs),
                total=len(inputs)
        )
    )
