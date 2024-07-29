#!/usr/bin/env python
import torch.nn.functional as F
import torch

import os
os.environ["OMP_NUM_THREADS"] = "1"

from utils.multiprocess_utils import chunked_multiprocess_run
import random
import traceback
import json
from resemblyzer import VoiceEncoder
from tqdm import tqdm
from utils.binarizer_utils import get_mel2ph, get_pitch, build_phone_encoder
from utils.hparams import set_hparams, hparams
import numpy as np
from utils.indexed_datasets import IndexedDatasetBuilder
from modules.vocoders.registry import VOCODERS
import pandas as pd


class BinarizationError(Exception):
    pass


class BaseBinarizer:
    """
        Base class for data processing.
        1. *process* and *process_data_split*:
            process entire data, generate the train-test split (support parallel processing);
        2. *process_item*:
            process singe piece of data;
        3. *get_pitch*:
            infer the pitch using some algorithm;
        4. *get_align*:
            get the alignment using 'mel2ph' format (see https://arxiv.org/abs/1905.09263).
        5. phoneme encoder, voice encoder, etc.

        Subclasses should define:
        1. *load_metadata*:
            how to read multiple datasets from files;
        2. *train_item_names*, *valid_item_names*, *test_item_names*:
            how to split the dataset;
        3. load_ph_set:
            the phoneme set.
    """

    def __init__(self, processed_data_dir=None):
        if processed_data_dir is None:
            processed_data_dir = hparams['processed_data_dir']   # 获得模型中yaml中的processed_data_dir即data/processed/比如（ljspeech）
        self.processed_data_dirs = processed_data_dir.split(",")  # 如果多个路径，则分别分隔开，
        self.binarization_args = hparams['binarization_args']    # 获取shuffle等
        self.item2txt = {}
        self.item2ph = {}
        self.item2wavfn = {}
        self.item2tgfn = {}
        self.item2spk = {}
        for ds_id, processed_data_dir in enumerate(self.processed_data_dirs):
            self.meta_df = pd.read_csv(f"{processed_data_dir}/metadata_phone.csv", dtype=str)  # data/processed/ljspeech/metadata_phone.csv
            for r_idx, r in self.meta_df.iterrows():  # 可打开metadata_phone.csv文件看看
                item_name = raw_item_name = r['item_name']  # item_name = LJ0**-****
                if len(self.processed_data_dirs) > 1: # 基本上用不到，就一个目录，如果多个就item_name =*_LJ0**-****
                    item_name = f'ds{ds_id}_{item_name}'
                self.item2txt[item_name] = r['txt']  # 文件中的文本，即语音或歌词文本
                self.item2ph[item_name] = r['ph']    # 文件中的音素，

                #raw_data_dir: 'data/raw/LJSpeech-1.1'  # item2wavfn[item_name]= 'data/raw/LJSpeech-1.1/wavs/LJ0**-0***.wav'
                self.item2wavfn[item_name] = os.path.join(hparams['raw_data_dir'], 'wavs', os.path.basename(r['wav_fn']).split('_')[1])
                self.item2spk[item_name] = r.get('spk', 'SPK1')  # item2spk[item_name]=SPK1
                if len(self.processed_data_dirs) > 1: # 忽略
                    self.item2spk[item_name] = f"ds{ds_id}_{self.item2spk[item_name]}"

        self.item_names = sorted(list(self.item2txt.keys()))  # 先获取键值，keys='LJ001-0001'，在转换成列表后排序
        if self.binarization_args['shuffle']:   # shuffle: false
            random.seed(1234)                   # 固定的随机种子
            random.shuffle(self.item_names)

    @property
    def train_item_names(self):
        return self.item_names[hparams['test_num']+hparams['valid_num']:]  # test_num: 523 ,valid_num: 348
        # 返回训练集为523+348到最大

    @property
    def valid_item_names(self):
        return self.item_names[0: hparams['test_num']+hparams['valid_num']]  # # test_num: 523 ,valid_num: 348
        # 返回有效集为前523+348个

    @property
    def test_item_names(self):
        return self.item_names[0: hparams['test_num']]  # Audios for MOS testing are in 'test_ids'
        # 返回测试集为前523个

    def build_spk_map(self):
        spk_map = set()   # set() 函数创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可以计算交集、差集、并集等
        for item_name in self.item_names:
            spk_name = self.item2spk[item_name]
            spk_map.add(spk_name)
        spk_map = {x: i for i, x in enumerate(sorted(list(spk_map)))}
        # spk_map = {x: i for i, x in enumerate(spk_map)}
        assert len(spk_map) == 0 or len(spk_map) <= hparams['num_spk'], len(spk_map)  # num_spk: 1
        return spk_map  # spk_map={'SPK1': 0}

    def item_name2spk_id(self, item_name):
        return self.spk_map[self.item2spk[item_name]]  # item_name2spk_id=0

    def _phone_encoder(self):
        ph_set_fn = f"{hparams['binary_data_dir']}/phone_set.json"  # 'data/binary/ljspeech/phone_set.json'
        ph_set = []
        if hparams['reset_phone_dict'] or not os.path.exists(ph_set_fn):   # reset_phone_dict: true
            for processed_data_dir in self.processed_data_dirs:
                ph_set += [x.split(' ')[0] for x in open(f'{processed_data_dir}/dict.txt').readlines()]
                # ['!', ',', '.', ';', '<BOS>', '<EOS>', '?', 'AA0', 'AA1', 'AA2', 'AE0', 'AE1', .....

            ph_set = sorted(set(ph_set))  # set()除去重复的词并重新排序。
            json.dump(ph_set, open(ph_set_fn, 'w'))  # 写入.json文件中。
        else:
            ph_set = json.load(open(ph_set_fn, 'r'))  # 如果不用重新设置音素字典或者音素设置函数已经存在，则直接加载读取即可。
        print("| phone set: ", ph_set)   # 打印除phone set
        return build_phone_encoder(hparams['binary_data_dir'])  # 返回编码后结果


    # 获取有效集或者测试集或者训练集里的item_name, ph, txt, tg_fn, wav_fn, spk_id
    def meta_data(self, prefix):
        if prefix == 'valid':
            item_names = self.valid_item_names
        elif prefix == 'test':
            item_names = self.test_item_names
        else:
            item_names = self.train_item_names
        for item_name in item_names:
            ph = self.item2ph[item_name]
            txt = self.item2txt[item_name]
            tg_fn = self.item2tgfn.get(item_name)
            wav_fn = self.item2wavfn[item_name]
            spk_id = self.item_name2spk_id(item_name)
            yield item_name, ph, txt, tg_fn, wav_fn, spk_id

    def process(self):
        os.makedirs(hparams['binary_data_dir'], exist_ok=True)   # os.makedirs() 方法用于递归创建目录 binary_data_dir:'data/binary/ljspeech',
                                                                 # exist_ok参数设置为True时，可以自动判断当文件夹已经存在就不创建
        self.spk_map = self.build_spk_map()
        print("| spk_map: ", self.spk_map)  # | spk_map:  {'SPK1': 0}
        spk_map_fn = f"{hparams['binary_data_dir']}/spk_map.json"
        json.dump(self.spk_map, open(spk_map_fn, 'w'))   # 存入 data/binary/ljspeech/spk_map.json中

        self.phone_encoder = self._phone_encoder()   # 编码
        self.process_data('valid')
        self.process_data('test')
        self.process_data('train')

    def process_data(self, prefix):
        data_dir = hparams['binary_data_dir']  # binary_data_dir: 'data/binary/ljspeech'
        args = []
        builder = IndexedDatasetBuilder(f'{data_dir}/{prefix}')  # 将此路径文件内容序列化
        #builder = IndexedDatasetBuilder(data_dir, prefix=prefix, allowed_attr=self.data_attrs)
        lengths = []
        f0s = []
        total_sec = 0
        if self.binarization_args['with_spk_embed']:  # with_spk_embed: true
            voice_encoder = VoiceEncoder().cuda()  # 语音编码器

        meta_data = list(self.meta_data(prefix))  # 将其转换成列表
        # print("meta_data: ", meta_data)
        # print("self.phone_encoder ", self.phone_encoder)
        # print("self.binarization_args ", self.binarization_args)
        for m in meta_data:
            args.append(list(m) + [self.phone_encoder, self.binarization_args])
        num_workers = int(os.getenv('N_PROC', os.cpu_count() // 3))  # 为了多进程处理数据
        for f_id, (_, item) in enumerate(
                zip(tqdm(meta_data), chunked_multiprocess_run(self.process_item, args, num_workers=num_workers))):
            if item is None:
                continue
            # print("================",item['wav'])
            item['spk_embed'] = voice_encoder.embed_utterance(item['wav']) \
                if self.binarization_args['with_spk_embed'] else None
            if not self.binarization_args['with_wav'] and 'wav' in item:   # 判断wav是否在item中和with是否为false
                print("del wav")
                del item['wav']
            builder.add_item(item)   # 添加序列化item
            lengths.append(item['len'])  # 添加长度
            total_sec += item['sec']
            if item.get('f0') is not None:
                f0s.append(item['f0'])
        builder.finalize()
        np.save(f'{data_dir}/{prefix}_lengths.npy', lengths)  # 保存为lengths.npy
        if len(f0s) > 0:
            f0s = np.concatenate(f0s, 0)  # 一次完成多个数组的拼接
            f0s = f0s[f0s != 0]
            np.save(f'{data_dir}/{prefix}_f0s_mean_std.npy', [np.mean(f0s).item(), np.std(f0s).item()])  # 保存均值方差
        print(f"| {prefix} total duration: {total_sec:.3f}s")  # | test total duration: 3445.783s

    @classmethod
    def process_item(cls, item_name, ph, txt, tg_fn, wav_fn, spk_id, encoder, binarization_args):
        if hparams['vocoder'] in VOCODERS:  # vocoder: pwg
            wav, mel = VOCODERS[hparams['vocoder']].wav2spec(wav_fn)
        else:
            wav, mel = VOCODERS[hparams['vocoder'].split('.')[-1]].wav2spec(wav_fn)
        res = {
            'item_name': item_name, 'txt': txt, 'ph': ph, 'mel': mel, 'wav': wav, 'wav_fn': wav_fn,
            'sec': len(wav) / hparams['audio_sample_rate'], 'len': mel.shape[0], 'spk_id': spk_id
        }
        try:
            if binarization_args['with_f0']:  # with_f0: true
                cls.get_pitch(wav, mel, res)
            if binarization_args['with_txt']:
                try:
                    phone_encoded = res['phone'] = encoder.encode(ph)
                except:
                    traceback.print_exc()
                    raise BinarizationError(f"Empty phoneme")
                if binarization_args['with_align']:
                    cls.get_align(tg_fn, ph, mel, phone_encoded, res)
        except BinarizationError as e:
            print(f"| Skip item ({e}). item_name: {item_name}, wav_fn: {wav_fn}")
            return None
        return res

    @staticmethod
    def get_align(tg_fn, ph, mel, phone_encoded, res):
        if tg_fn is not None and os.path.exists(tg_fn):
            mel2ph, dur = get_mel2ph(tg_fn, ph, mel, hparams)
        else:
            raise BinarizationError(f"Align not found")
        if mel2ph.max() - 1 >= len(phone_encoded):
            raise BinarizationError(
                f"Align does not match: mel2ph.max() - 1: {mel2ph.max() - 1}, len(phone_encoded): {len(phone_encoded)}")
        res['mel2ph'] = mel2ph
        res['dur'] = dur

    @staticmethod
    def get_pitch(wav, mel, res):
        f0, pitch_coarse = get_pitch(wav, mel, hparams)
        if sum(f0) == 0:
            raise BinarizationError("Empty f0")
        res['f0'] = f0
        res['pitch'] = pitch_coarse



if __name__ == "__main__":
    set_hparams()
    BaseBinarizer().process()
