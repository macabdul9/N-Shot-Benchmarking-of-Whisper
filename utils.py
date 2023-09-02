import re
import jiwer
from datasets import load_dataset, DatasetDict, ReadInstruction, load_from_disk, IterableDatasetDict
import jiwer 
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import os
import json
from pathlib import Path

MGB2_EXAMPLES = 376011 # https://huggingface.co/datasets/arbml/mgb2_speech/blob/main/README.md
SEED = 42


def preprocessor(text):
        # Based on https://arxiv.org/pdf/2105.14779.pdf

        # 1- remove the sepecial cahrachters and diacritics
        #   -- We did NOT convert the Latin characters to lower case as we are not working on Code-Switch dataset, so we removed all the Latin letters.
        text = re.sub(r"[^0-9\u0621-\u064A\u0660-\u0669%@\s]", "", text)
        # text = re.sub(r"[^0-9\u0600-\u06FF\u0660-\u0669%@\s]", "", text) #065E # include diacritics

        # 2- transliterating all Arabic digits to Arabic numerals (We used Whisper to verify the numerical output)
        text = re.sub(r"[٠-٩]",lambda x: str(int(x.group())), text)

        # 3- Normalize alefs
        text = re.sub("[إأٱآا]", "ا", text)

        # For Haa nad Taa, we didn't see a problem in CV9.0 data. (We need to discuss this further)

        # - Remove extra spaces
        text = " ".join(text.split())

        return text
    

def add_path(example, dataset_dir):
    example["audio"] = os.path.join(dataset_dir, "clips", example["audio"])
    return example
    
def data_loader(load_from_local, dataset_dir, cache_dir="./cache", dataset_name=None):
    
    if load_from_local:
        # load data from local
        dataset = load_dataset(
            'csv', 
            # dataset_dir=dataset_dir, 
            data_files={
                "train":os.path.join(dataset_dir, "train.tsv"), 
                "dev":os.path.join(dataset_dir, "dev.tsv"), 
                "test":os.path.join(dataset_dir, "test.tsv")}, 
            delimiter="\t",
            cache_dir=cache_dir
        )
        
        # add complete path to audio file
        for split in dataset.keys():
            dataset[split] = dataset[split].map(lambda example: add_path(example, dataset_dir))
            
        return dataset
        
    elif "common_voice" in dataset_dir:
        # load data from huggingface datasets
        dataset = DatasetDict()
        dataset['train'] = load_dataset(dataset_dir, "ar", split="train", use_auth_token=True, cache_dir=cache_dir, ignore_verifications=False)
        dataset['test'] = load_dataset(dataset_dir, "ar", split="test", use_auth_token=True, cache_dir=cache_dir, ignore_verifications=False)
        dataset['dev'] = load_dataset(dataset_dir, "ar", split="validation", use_auth_token=True, cache_dir=cache_dir, ignore_verifications=False)
        
        # rename text coloums to match the format of local data
        for split in dataset.keys():
            dataset[split] = dataset[split].rename_column("sentence", "text")
            # dataset[split] = dataset[split].rename_column("path", "audio")
        
            
        return dataset
    
    elif "fleurs" in dataset_dir:
        # load data from huggingface datasets
        dataset = DatasetDict()
        dataset['train'] = load_dataset(dataset_dir, "ar_eg", split="train", use_auth_token=True, cache_dir=cache_dir)
        dataset['test'] = load_dataset(dataset_dir, "ar_eg", split="test", use_auth_token=True, cache_dir=cache_dir)
        dataset['dev'] = load_dataset(dataset_dir, "ar_eg", split="validation", use_auth_token=True, cache_dir=cache_dir)
        
        # rename text coloums to match the format of local data
        # rename text coloums to match the format of local data
        for split in dataset.keys():
            dataset[split] = dataset[split].rename_column("transcription", "text")
            # dataset[split] = dataset[split].rename_column("path", "audio")
    
        return dataset

    else:
        return load_dataset(dataset_dir, use_auth_token=True, cache_dir=cache_dir)
                

def sampler(dataset, dataset_duration, training_duration):
    
    pct = training_duration / dataset_duration
    
    dataset = dataset.shuffle().select(range(int(pct * len(dataset))))
    
    return dataset
    
    
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

def write_txt_file(ordered_tgt, path):
    f = Path(path).open("w")
    for ln in ordered_tgt:
        f.write(ln + "\n")
        f.flush()
        
def save_json(content, path, indent=4, **json_dump_kwargs):
    with open(path, "w") as f:
        json.dump(content, f, indent=indent, sort_keys=True, **json_dump_kwargs)
    
def lmap(f, x) -> List:
    """list(map(f, x))"""
    return list(map(f, x))



if __name__=="__main__":
    
    
    # # import librosa
    
    # # from tqdm import tqdm
    
    # # load mgb2 
    
    # # dataset = mgb2_loader(training_duration=1)
    
    # # print(dataset)
    
    # # features = list(next(iter(dataset.values())).features.keys())
    
    # # print(features)
    
    
    
    
    # # for _, batch in tqdm(enumerate(dataset['train'])):
        
    # #     print(batch)
    # #     break
    
    
    # # train_durations = [librosa.get_duration(y=batch['audio']['array']) for _, batch in tqdm(enumerate(dataset['train']))]
    
    
    # # print(sum(train_durations)/(60*60))
    
    # # batch = next(iter(dataset['train']))
    
    # # print(batch)
    
    # # print(batch['audio']['array'])
    
    
    

    # # If the dataset is gated/private, make sure you have run huggingface-cli login
    # # dataset = load_dataset("arbml/mgb3", use_auth_token=True, streaming=False, cache_dir="./cache/", ignore_verifications=True)
    
    # # raw_datasets_features = list(next(iter(dataset.values())).features.keys())
    
    # # print(dataset)
    
    # # print(raw_datasets_features)
    
    # # dataset = load_dataset("arbml/mgb2_speech", split="validation", use_auth_token=True, streaming=False, cache_dir="$SLURM_TMPRDIR/cache/", ignore_verifications=True)
    
    # # print(dataset)
    
    # import datasets
    
    # # dataset = datasets.load_from_disk("/home/awaheed/scratch/ASRs/data/mgb2.hf")
    
    # dataset = datasets.load_from_disk("/home/awaheed/scratch/ASRs/whisper-experiments/mgb2_0/4.hf")
    
    # print(dataset)
    
    # # print(dataset['test'])
    pass
    


    
    
    