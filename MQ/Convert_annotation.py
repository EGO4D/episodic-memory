import json
import torch
import os

######################################################################################################
#                     Load data
######################################################################################################
annotation_path = "../../annotations/"  # Change to your own path containing canonical annotation files
feat_path = "video_features/"  # Change to your own path containing features of canonical videos
info_path = annotation_path + 'ego4d.json'
annot_path_train = annotation_path + 'moments_train.json'
annot_path_val = annotation_path + 'moments_val.json'
annot_path_test = annotation_path + 'moments_test_unannotated.json'

with open(annot_path_train, 'r') as f:
    v_annot_train = json.load(f)

with open(annot_path_val, 'r') as f:
    v_annot_val = json.load(f)

with open(annot_path_test, 'r') as f:
    v_annot_test = json.load(f)

with open(info_path, 'r') as f:
    feat_info=json.load(f)

v_all_duration = {}
for video in feat_info['videos']:
    v_id = video['video_uid']
    v_dur = video['duration_sec']
    v_all_duration[v_id] = v_dur

v_annot = {}
v_annot['videos'] = v_annot_train['videos'] + v_annot_val['videos'] + v_annot_test['videos']

######################################################################################################
#                     Convert video annotations to clip annotations: clip_annot_1
######################################################################################################
clip_annot_1 = {}
for video in v_annot['videos']:
    vid = video['video_uid']
    clips = video['clips']
    v_duration = v_all_duration[vid] #feat_info[feat_info.video_uid == vid].canonical_video_duration_sec.values[0]
    try:
        feats = torch.load(os.path.join(feat_path, vid + '.pt'))
    except:
        print(f'{vid} features do not exist!')
        continue
    fps = feats.shape[0] / v_duration
    for clip in clips:
        clip_id = clip['clip_uid']

        if clip_id not in clip_annot_1.keys():
            clip_annot_1[clip_id] = {}
            clip_annot_1[clip_id]['video_id'] = vid
            clip_annot_1[clip_id]['clip_id'] = clip_id
            clip_annot_1[clip_id]['parent_start_sec'] = clip['video_start_sec']
            clip_annot_1[clip_id]['parent_end_sec'] = clip['video_end_sec']
            clip_annot_1[clip_id]['v_duration'] = v_duration
            clip_annot_1[clip_id]['fps'] = fps
            clip_annot_1[clip_id]['annotations'] = []
            clip_annot_1[clip_id]['subset'] = video['split']

        if video['split'] != 'test':
            annotations = clip['annotations']
            for cnt, annot in enumerate(annotations):
                for label in annot['labels']:
                    if label['primary']:
                        clip_annot_1[clip_id]['annotations'].append(label)

#######################################################################
## If there are no remaining annotations for a clip ###
## Remove the clip ###
remove_list = []
for k, v in clip_annot_1.items():
    if v['subset']!='test' and len(v['annotations']) == 0:
        print(f'NO annotations: video {k}')
        remove_list.append(k)

for item in remove_list:
    del clip_annot_1[item]

cnt_train = 0
cnt_val = 0
for k, v in clip_annot_1.items():
    if v['subset'] == 'train':
        cnt_train += 1
    elif v['subset'] == 'val':
        cnt_val += 1

print(f"Number of clips in training: {cnt_train}")
print(f"Number of clips in validation: {cnt_val}")

with open("Evaluation/ego4d/annot/clip_annotations.json", "w") as fp:
    json.dump(clip_annot_1, fp)




