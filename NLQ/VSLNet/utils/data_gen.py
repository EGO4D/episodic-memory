import codecs
import multiprocessing
import os
from collections import Counter
from typing import List, Optional

import numpy as np
from nltk.tokenize import word_tokenize
from tqdm import tqdm

from utils.data_util import (
    load_json,
    load_lines,
    load_pickle,
    save_pickle,
    time_to_index,
)

PAD, UNK = "<PAD>", "<UNK>"


class EpisodicNLQProcessor:
    def __init__(self, remove_empty_queries_from: Optional[List[str]]):
        super(EpisodicNLQProcessor, self).__init__()
        self.idx_counter = 0
        self.remove_empty_queries_from = (
            set()
            if remove_empty_queries_from is None else
            set(remove_empty_queries_from)
        )

    def reset_idx_counter(self):
        self.idx_counter = 0

    def process_data_tan(self, data, scope):
        skipped = 0
        results = []
        for vid, data_item in tqdm(
            data.items(), total=len(data), desc=f"process episodic nlq {scope}"
        ):
            fps = float(data_item["fps"])
            duration = float(data_item["num_frames"]) / fps
            zipper = zip(
                data_item["timestamps"],
                data_item["exact_times"],
                data_item["sentences"],
                data_item["annotation_uids"],
                data_item["query_idx"],
            )
            for timestamp, exact_time, sentence, ann_uid, query_idx in zipper:
                start_time = max(0.0, float(timestamp[0]) / fps)
                end_time = min(float(timestamp[1]) / fps, duration)
                if self._predictor != "bert":
                    words = word_tokenize(sentence.strip().lower(), language="english")
                else:
                    words = sentence
                record = {
                    "sample_id": self.idx_counter,
                    "vid": str(vid),
                    "s_time": start_time,
                    "e_time": end_time,
                    "exact_s_time": exact_time[0],
                    "exact_e_time": exact_time[1],
                    "duration": duration,
                    "words": words,
                    "query": sentence.strip().lower(),
                    "annotation_uid": ann_uid,
                    "query_idx": query_idx,
                }
                if (
                    abs(exact_time[0] - exact_time[1]) <= 1/30 and
                    scope in self.remove_empty_queries_from
                ):
                    skipped += 1
                    continue
                results.append(record)
                self.idx_counter += 1
        print(f"{scope}: skipped = {skipped}, remaining = {len(results)}")
        return results

    def convert(self, data_dir, predictor=None):
        self._predictor = predictor
        self.reset_idx_counter()
        if not os.path.exists(data_dir):
            raise ValueError("data dir {} does not exist".format(data_dir))
        # load raw data
        train_data = load_json(os.path.join(data_dir, "train.json"))
        val_data = load_json(os.path.join(data_dir, "val.json"))
        test_data = load_json(os.path.join(data_dir, "test.json"))

        # process data
        train_set = self.process_data_tan(train_data, scope="train")
        val_set = self.process_data_tan(val_data, scope="val")
        test_set = self.process_data_tan(test_data, scope="test")
        return train_set, val_set, test_set


def load_glove(glove_path):
    vocab = list()
    with codecs.open(glove_path, mode="r", encoding="utf-8") as f:
        for line in tqdm(f, total=2196018, desc="load glove vocabulary"):
            line = line.lstrip().rstrip().split(" ")
            if len(line) == 2 or len(line) != 301:
                continue
            word = line[0]
            vocab.append(word)
    return set(vocab)


def filter_glove_embedding(word_dict, glove_path):
    vectors = np.zeros(shape=[len(word_dict), 300], dtype=np.float32)
    with codecs.open(glove_path, mode="r", encoding="utf-8") as f:
        for line in tqdm(f, total=2196018, desc="load glove embeddings"):
            line = line.lstrip().rstrip().split(" ")
            if len(line) == 2 or len(line) != 301:
                continue
            word = line[0]
            if word in word_dict:
                vector = [float(x) for x in line[1:]]
                word_index = word_dict[word]
                vectors[word_index] = np.asarray(vector)
    return np.asarray(vectors)


def vocab_emb_gen(datasets, emb_path):
    # generate word dict and vectors
    emb_vocab = load_glove(emb_path)
    word_counter, char_counter = Counter(), Counter()
    for data in datasets:
        for record in data:
            for word in record["words"]:
                word_counter[word] += 1
                for char in list(word):
                    char_counter[char] += 1
    word_vocab = list()
    for word, _ in word_counter.most_common():
        if word in emb_vocab:
            word_vocab.append(word)
    tmp_word_dict = dict([(word, index) for index, word in enumerate(word_vocab)])
    vectors = filter_glove_embedding(tmp_word_dict, emb_path)
    word_vocab = [PAD, UNK] + word_vocab
    word_dict = dict([(word, idx) for idx, word in enumerate(word_vocab)])
    # generate character dict
    char_vocab = [PAD, UNK] + [
        char for char, count in char_counter.most_common() if count >= 5
    ]
    char_dict = dict([(char, idx) for idx, char in enumerate(char_vocab)])
    return word_dict, char_dict, vectors


def dataset_gen(
    data, vfeat_lens, word_dict, char_dict, max_pos_len, scope, num_workers=1
):
    # Worker method for multiprocessing.
    def worker(
        worker_data,
        vfeat_lens,
        word_dict,
        char_dict,
        max_pos_len,
        scope,
        worker_id,
        output_q,
    ):
        worker_dataset = list()
        description = f"process {scope} data [{worker_id}]"
        for record in tqdm(worker_data, total=len(worker_data), desc=description):
            vid = record["vid"]
            if vid not in vfeat_lens:
                continue
            s_ind, e_ind, _ = time_to_index(
                record["s_time"], record["e_time"], vfeat_lens[vid], record["duration"]
            )

            word_ids, char_ids = [], []
            for word in record["words"][0:max_pos_len]:
                word_id = word_dict[word] if word in word_dict else word_dict[UNK]
                char_id = [
                    char_dict[char] if char in char_dict else char_dict[UNK]
                    for char in word
                ]
                word_ids.append(word_id)
                char_ids.append(char_id)
            result = {
                "sample_id": record["sample_id"],
                "vid": record["vid"],
                "s_time": record["s_time"],
                "e_time": record["e_time"],
                "duration": record["duration"],
                "words": record["words"],
                "query": record["query"],
                "s_ind": int(s_ind),
                "e_ind": int(e_ind),
                "v_len": vfeat_lens[vid],
                "w_ids": word_ids,
                "c_ids": char_ids,
                "annotation_uid": record["annotation_uid"],
                "query_idx": record["query_idx"],
            }
            worker_dataset.append(result)
        output_q.put({worker_id: worker_dataset})

    # Multithread version.
    output_q = multiprocessing.Queue()
    jobs = []
    chunk_size = len(data) // num_workers + 1
    for worker_id in range(num_workers):
        allotment = data[worker_id * chunk_size : (worker_id + 1) * chunk_size]
        inputs = (allotment, vfeat_lens, word_dict, char_dict, max_pos_len, scope)
        inputs += (worker_id, output_q)
        process = multiprocessing.Process(target=worker, args=inputs)
        jobs.append(process)
        process.start()

    # Wait for all the jobs to finish and collect the output.
    collated_results = {}
    for _ in jobs:
        collated_results.update(output_q.get())
    for job in jobs:
        job.join()

    # Flatten and sort the dataset.
    dataset = []
    sorted_worker_id = sorted(collated_results.keys())
    for worker_id in sorted_worker_id:
        dataset.extend(collated_results[worker_id])
    return dataset


def dataset_gen_bert(data, vfeat_lens, tokenizer, max_pos_len, scope, num_workers=1):
    # Worker method for multiprocessing.
    def worker(
        worker_data, vfeat_lens, tokenizer, max_pos_len, scope, worker_id, output_q
    ):
        worker_dataset = list()
        description = f"process {scope} data [{worker_id}]"
        for record in tqdm(worker_data, total=len(worker_data), desc=description):
            vid = record["vid"]
            if vid not in vfeat_lens:
                continue
            s_ind, e_ind, _ = time_to_index(
                record["s_time"], record["e_time"], vfeat_lens[vid], record["duration"]
            )
            word_ids = tokenizer(record["query"])
            result = {
                "sample_id": record["sample_id"],
                "vid": record["vid"],
                "s_time": record["s_time"],
                "e_time": record["e_time"],
                "duration": record["duration"],
                "words": record["words"],
                "query": record["query"],
                "s_ind": int(s_ind),
                "e_ind": int(e_ind),
                "v_len": vfeat_lens[vid],
                "w_ids": word_ids,
                "annotation_uid": record["annotation_uid"],
                "query_idx": record["query_idx"],
            }
            worker_dataset.append(result)
        output_q.put({worker_id: worker_dataset})

    # Multithread version.
    output_q = multiprocessing.Queue()
    jobs = []
    chunk_size = len(data) // num_workers + 1
    for worker_id in range(num_workers):
        allotment = data[worker_id * chunk_size : (worker_id + 1) * chunk_size]
        inputs = (allotment, vfeat_lens, tokenizer, max_pos_len, scope)
        inputs += (worker_id, output_q)
        process = multiprocessing.Process(target=worker, args=inputs)
        jobs.append(process)
        process.start()

    # Wait for all the jobs to finish and collect the output.
    collated_results = {}
    for _ in jobs:
        collated_results.update(output_q.get())
    for job in jobs:
        job.join()

    # Flatten and sort the dataset.
    dataset = []
    sorted_worker_id = sorted(collated_results.keys())
    for worker_id in sorted_worker_id:
        dataset.extend(collated_results[worker_id])
    return dataset


def gen_or_load_dataset(configs):
    if not os.path.exists(configs.save_dir):
        os.makedirs(configs.save_dir)
    data_dir = os.path.join("data", "dataset", configs.task)
    feature_dir = os.path.join("data", "features", configs.task, configs.fv)
    if configs.suffix is None:
        save_path = os.path.join(
            configs.save_dir,
            "_".join(
                [configs.task, configs.fv, str(configs.max_pos_len), configs.predictor]
            )
            + ".pkl",
        )
    else:
        save_path = os.path.join(
            configs.save_dir,
            "_".join(
                [configs.task, configs.fv, str(configs.max_pos_len), configs.suffix]
            )
            + ".pkl",
        )
    if os.path.exists(save_path):
        print(f"Loading data from existing save path {save_path}", flush=True)
        dataset = load_pickle(save_path)
        return dataset
    print("Generating data for dataloader", flush=True)
    feat_len_path = os.path.join(feature_dir, "feature_shapes.json")
    emb_path = os.path.join("data", "features", "glove.840B.300d.txt")
    # load video feature length
    vfeat_lens = load_json(feat_len_path)
    for vid, vfeat_len in vfeat_lens.items():
        vfeat_lens[vid] = min(configs.max_pos_len, vfeat_len)
    # load data
    processor = EpisodicNLQProcessor(configs.remove_empty_queries_from)

    train_data, val_data, test_data = processor.convert(
        data_dir, predictor=configs.predictor
    )
    # generate dataset
    data_list = (
        [train_data, test_data]
        if val_data is None
        else [train_data, val_data, test_data]
    )
    if configs.predictor == "bert":
        from transformers import BertTokenizer, BertForPreTraining

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        train_set = dataset_gen_bert(
            train_data,
            vfeat_lens,
            tokenizer,
            configs.max_pos_len,
            "train",
            num_workers=configs.num_workers,
        )
        if val_data:
            val_set = dataset_gen_bert(
                val_data,
                vfeat_lens,
                tokenizer,
                configs.max_pos_len,
                "val",
                num_workers=configs.num_workers,
            )
        else:
            val_set = None
        test_set = dataset_gen_bert(
            test_data,
            vfeat_lens,
            tokenizer,
            configs.max_pos_len,
            "test",
            num_workers=configs.num_workers,
        )
        n_val = 0 if val_set is None else len(val_set)
        dataset = {
            "train_set": train_set,
            "val_set": val_set,
            "test_set": test_set,
            "n_train": len(train_set),
            "n_val": n_val,
            "n_test": len(test_set),
        }
    else:
        word_dict, char_dict, vectors = vocab_emb_gen(data_list, emb_path)
        train_set = dataset_gen(
            train_data,
            vfeat_lens,
            word_dict,
            char_dict,
            configs.max_pos_len,
            "train",
            num_workers=configs.num_workers,
        )
        if val_data:
            val_set = dataset_gen(
                val_data,
                vfeat_lens,
                word_dict,
                char_dict,
                configs.max_pos_len,
                "val",
                num_workers=configs.num_workers,
            )
        else:
            val_set = None
        test_set = dataset_gen(
            test_data,
            vfeat_lens,
            word_dict,
            char_dict,
            configs.max_pos_len,
            "test",
            num_workers=configs.num_workers,
        )
        # save dataset
        n_val = 0 if val_set is None else len(val_set)
        dataset = {
            "train_set": train_set,
            "val_set": val_set,
            "test_set": test_set,
            "word_dict": word_dict,
            "char_dict": char_dict,
            "word_vector": vectors,
            "n_train": len(train_set),
            "n_val": n_val,
            "n_test": len(test_set),
            "n_words": len(word_dict),
            "n_chars": len(char_dict),
        }
    save_pickle(dataset, save_path)
    return dataset
