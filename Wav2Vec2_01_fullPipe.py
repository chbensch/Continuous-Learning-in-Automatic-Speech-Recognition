import os
import torchaudio
import datasets 
import librosa
import numpy as np
import re
import torch

from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor


def remove_special_characters(batch):
    chars_to_keep = '[^A-Za-zäüö ]+'
    batch["sentence"] = re.sub(chars_to_keep, '', batch["sentence"]).lower() + " "
    return batch

def speech_file_to_array_subsample_fn(batch):  
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = speech_array
    batch["speech"] = librosa.resample(np.asarray(batch["speech"]), sampling_rate, 16_000)
    batch["sampling_rate"] = 16_000
    batch["target_text"] = batch["sentence"]
    return batch

def pre_process(datahouse):

    batch_size = 2500
    start = batch_size
    end = len(datahouse["path"])
    print("total length: " + str(end))
    process = True

    result_batch = datasets.from_dict(datahouse[0:batch_size])
    result_batch = result_batch.map(speech_file_to_array_subsample_fn, remove_columns=result_batch.column_names, num_proc=4) 

    lenSBS = len(datahouse["path"])

    print("len of small batch start: " + str(lenSBS))

    while (process):
        i = start+batch_size
        if (i < end):
            print("From: " + str(start) + " to: " + str(i))
            small_batch = datasets.from_dict(datahouse[start:i])
            start = start + batch_size
        else:
            print("LAST BATCH!")
            print("From: " + str(start) + " to: " + str(end))
            small_batch = datasets.from_dict(datahouse[start:end])
            process = False

        small_batch = small_batch.map(speech_file_to_array_subsample_fn, remove_columns=small_batch.column_names, num_proc=4)
        result_batch = datasets.concatenate_datasets([result_batch, small_batch]) 
    
    return result_batch

common_voice_train = datasets.load_dataset("common_voice", "de", split="train")
#common_voice_validation = load_dataset("common_voice", "de", split="validation")
#common_voice_test = load_dataset("common_voice", "de", split="test")


common_voice_train = common_voice_train.map(remove_special_characters)
#common_voice_validation = common_voice_validation.map(remove_special_characters)
#common_voice_test = common_voice_test.map(remove_special_characters)

tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

common_voice_train = pre_process(common_voice_train)


common_voice_train.save_to_disk("./common_voice_train")