import numpy as np
import torch
import torch.utils.data

from utils.data_util import pad_seq, pad_char_seq, pad_video_seq


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, video_features):
        super(Dataset, self).__init__()
        self.dataset = dataset
        self.video_features = video_features

    def __getitem__(self, index):
        record = self.dataset[index]
        video_feature = self.video_features[record["vid"]]
        s_ind, e_ind = int(record["s_ind"]), int(record["e_ind"])
        word_ids = record["w_ids"]
        char_ids = record.get("c_ids", None)
        return record, video_feature, word_ids, char_ids, s_ind, e_ind

    def __len__(self):
        return len(self.dataset)


def train_collate_fn(data):
    records, video_features, word_ids, char_ids, s_inds, e_inds = zip(*data)
    # If BERT is used, pad individual components of the dictionary.
    if not isinstance(word_ids[0], list):
        pad_input_ids, _ = pad_seq([ii["input_ids"] for ii in word_ids])
        pad_attention_mask, _ = pad_seq([ii["attention_mask"] for ii in word_ids])
        pad_token_type_ids, _ = pad_seq([ii["token_type_ids"] for ii in word_ids])
        word_ids = {
            "input_ids": torch.LongTensor(pad_input_ids),
            "attention_mask": torch.LongTensor(pad_attention_mask),
            "token_type_ids": torch.LongTensor(pad_token_type_ids),
        }
        char_ids = None
    else:
        # process word ids
        word_ids, _ = pad_seq(word_ids)
        word_ids = np.asarray(word_ids, dtype=np.int32)  # (batch_size, w_seq_len)
        # process char ids
        char_ids, _ = pad_char_seq(char_ids)
        char_ids = np.asarray(
            char_ids, dtype=np.int32
        )  # (batch_size, w_seq_len, c_seq_len)
        word_ids = torch.tensor(word_ids, dtype=torch.int64)
        char_ids = torch.tensor(char_ids, dtype=torch.int64)
    # process video features
    vfeats, vfeat_lens = pad_video_seq(video_features)
    vfeats = np.asarray(vfeats, dtype=np.float32)  # (batch_size, v_seq_len, v_dim)
    vfeat_lens = np.asarray(vfeat_lens, dtype=np.int32)  # (batch_size, )
    # process labels
    max_len = np.max(vfeat_lens)
    batch_size = vfeat_lens.shape[0]
    s_labels = np.asarray(s_inds, dtype=np.int64)
    e_labels = np.asarray(e_inds, dtype=np.int64)
    h_labels = np.zeros(shape=[batch_size, max_len], dtype=np.int32)
    extend = 0.1
    for idx in range(batch_size):
        st, et = s_inds[idx], e_inds[idx]
        cur_max_len = vfeat_lens[idx]
        extend_len = round(extend * float(et - st + 1))
        if extend_len > 0:
            st_ = max(0, st - extend_len)
            et_ = min(et + extend_len, cur_max_len - 1)
            h_labels[idx][st_ : (et_ + 1)] = 1
        else:
            h_labels[idx][st : (et + 1)] = 1
    # convert to torch tensor
    vfeats = torch.tensor(vfeats, dtype=torch.float32)
    vfeat_lens = torch.tensor(vfeat_lens, dtype=torch.int64)
    s_labels = torch.tensor(s_labels, dtype=torch.int64)
    e_labels = torch.tensor(e_labels, dtype=torch.int64)
    h_labels = torch.tensor(h_labels, dtype=torch.int64)
    return records, vfeats, vfeat_lens, word_ids, char_ids, s_labels, e_labels, h_labels


def test_collate_fn(data):
    records, video_features, word_ids, char_ids, *_ = zip(*data)
    # If BERT is used, pad individual components of the dictionary.
    if not isinstance(word_ids[0], list):
        pad_input_ids, _ = pad_seq([ii["input_ids"] for ii in word_ids])
        pad_attention_mask, _ = pad_seq([ii["attention_mask"] for ii in word_ids])
        pad_token_type_ids, _ = pad_seq([ii["token_type_ids"] for ii in word_ids])
        word_ids = {
            "input_ids": torch.LongTensor(pad_input_ids),
            "attention_mask": torch.LongTensor(pad_attention_mask),
            "token_type_ids": torch.LongTensor(pad_token_type_ids),
        }
        char_ids = None
    else:
        # process word ids
        word_ids, _ = pad_seq(word_ids)
        word_ids = np.asarray(word_ids, dtype=np.int32)  # (batch_size, w_seq_len)
        # process char ids
        char_ids, _ = pad_char_seq(char_ids)
        char_ids = np.asarray(
            char_ids, dtype=np.int32
        )  # (batch_size, w_seq_len, c_seq_len)
        word_ids = torch.tensor(word_ids, dtype=torch.int64)
        char_ids = torch.tensor(char_ids, dtype=torch.int64)
    # process video features
    vfeats, vfeat_lens = pad_video_seq(video_features)
    vfeats = np.asarray(vfeats, dtype=np.float32)  # (batch_size, v_seq_len, v_dim)
    vfeat_lens = np.asarray(vfeat_lens, dtype=np.int32)  # (batch_size, )
    # convert to torch tensor
    vfeats = torch.tensor(vfeats, dtype=torch.float32)
    vfeat_lens = torch.tensor(vfeat_lens, dtype=torch.int64)
    return records, vfeats, vfeat_lens, word_ids, char_ids


def get_train_loader(dataset, video_features, configs):
    train_set = Dataset(dataset=dataset, video_features=video_features)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=configs.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=configs.data_loader_workers,
        collate_fn=train_collate_fn,
    )
    return train_loader


def get_test_loader(dataset, video_features, configs):
    test_set = Dataset(dataset=dataset, video_features=video_features)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=configs.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=configs.data_loader_workers,
        collate_fn=test_collate_fn,
    )
    return test_loader
