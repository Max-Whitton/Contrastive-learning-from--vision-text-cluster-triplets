from pathlib import Path
from typing import Any, Tuple
from collections import Counter
import os
import glob
import itertools
import json
import csv
import random
import re
import shutil
import time
import cv2 as cv

from PIL import Image
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import os.path
import sys

from multimodal.multimodal_data_module import MultiModalDataset, \
    MultiModalDataModule, read_vocab, load_data, load_and_print_info, \
    PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN, \
    PAD_TOKEN_ID, UNK_TOKEN_ID, SOS_TOKEN_ID, EOS_TOKEN_ID, \
    IMAGE_H, IMAGE_W, multiModalDataset_collate_fn

from multimodal.utils import *

import spacy
import clip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


VOCAB_FILENAME = "multimodal/vocab.json"

class MultiModalSAYCamDatasetBabyFM(MultiModalDataset):
    """
    Dataset that returns paired image-utterances from our training split of SAYCam 
    """

    def __init__(self, data_path, vocab, transform):
        """
        - Train/val/test data is all the same .json file b/c we are going to only keep the vision encoder after training and 
        evaluate intermediate checkpoints on Labeled-S linear probe
        """
        super().__init__()

        self.data = self._load_data(data_path)

        self.vocab = vocab
        self.transform = transform

        #load tokenizer for LLaVA
        self.nlp = spacy.load(
        'en_core_web_sm',
            exclude=[
                'attribute_ruler', 'lemmatizer', 'ner',
                'senter', 'parser', 'tagger', 'tok2vec']
        )

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.data)

    #NEED TO IMPLEMENT THIS 
    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any, Any]:
        """
        Returns an image-utterance pair in tuple
        (img, utterance_idxs, utterance_length, raw_utterances)
        """

        # get utterance and image 
        utterance = self.data[idx]["caption"]
        img_path = self.data[idx]["image"]

        # tokenize caption the same way the vocab.json was created
        utterance_words = [token.text for token in self.nlp.tokenizer(utterance)]
        
        utterance_words = [SOS_TOKEN] + utterance_words + [EOS_TOKEN]
        utterance_length = len(utterance_words)

        #ensure lower case when going from token to id
        utterance_idxs = torch.tensor([self.vocab.get(
            word, UNK_TOKEN_ID) for word in utterance_words], dtype=torch.long)

        # Load from "image" field — detect video by extension
        if img_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            cap = cv.VideoCapture(img_path)
            total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
            center_frame = total_frames // 2
            cap.set(cv.CAP_PROP_POS_FRAMES, center_frame)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                print(f"Bad video: {img_path}, skipping to random example")
                return self.__getitem__(random.randint(0, len(self)-1))
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
        else:
            img = Image.open(img_path).convert("RGB")
        img = img.resize((IMAGE_W, IMAGE_H))

        # apply transforms
        if self.transform is not None:
            img = self.transform(img)

        return img, utterance_idxs, utterance_length, [utterance]

    def _load_data(self, data_path):
        """
        Format data based on .json LLaVA format 
        """
        if data_path.suffix == '.json':
            return self._process_json(data_path)
        else:
            raise NotImplementedError

    def _process_json(self, data_path):
        """
        - Process json dataset and format it for training
        """
        with open(data_path, "r") as f:
            data = json.load(f)
        
        # Transform to training format - convert to lowercase to match tokenization step
        formatted_data = {
            int(i): {
                "image": entry["video_path"],
                "caption": entry["audio_caption"],
            }
            for i, entry in enumerate(data)
            if entry.get("audio_caption")
        }

        return formatted_data



class MultiModalSAYCamDataModuleBabyFM(MultiModalDataModule):
    """
    A data module created for our train split of the SAYCam dataset
    """

    def __init__(self, args=None) -> None:
        print("made it here\n\n\n")
        super().__init__(args)

        #can save any args if needed here

    #no additional arguments for now
    @staticmethod 
    def add_additional_to_argparse(parser):
        return None

    @staticmethod
    def add_to_argparse(parser):
        parser = super(MultiModalSAYCamDataModuleBabyFM,
                       MultiModalSAYCamDataModuleBabyFM).add_to_argparse(parser)
        parser = MultiModalSAYCamDataModuleBabyFM.add_additional_to_argparse(parser)
        return parser

    #Nothing to do here 
    def prepare_data(self, *args, **kwargs) -> None:
        super().prepare_data(*args, **kwargs)

    # Read in vocab to pass to text encoder and dataset classes
    def read_vocab(self):
        return read_vocab(VOCAB_FILENAME)

    #Overriding this since we don't have eval datasets or the path names used in the parent class
    def setup(self, *args, **kwargs) -> None:
        print("Calling setup for Saycam train split dataset!")

        # read vocab
        vocab = self.read_vocab()

        # read and create image-text data splits (train/val/test)
        self.datasets = self.create_datasets(vocab)

        # read and create eval data splits (val/test) -> they use multiple val and test methods (loss and accuracy)
        # self.eval_datasets = self.create_eval_datasets(vocab)

    #Create train, val, test datasets. Dataloading and collate is handled in parent class
    def create_datasets(self, vocab):
        datasets = {}
        print("Creating datasets for Saycam train split data...")
        
        stage_splits = [("train", TRAIN_DATA_DIR, self.transform)]

        for split, data_path, transform in stage_splits:
            dataset = MultiModalSAYCamDatasetBabyFM(
                data_path,
                vocab,
                transform=transform,
            )
            datasets[split] = dataset

        return datasets

    




class MultiModalSAYCamLLaVADataset(MultiModalDataset):
    """
    Dataset that returns paired image-utterances from baby S of the SAYCam Dataset
    and data from LLaVA pretraining modified for baby training. 
    """

    def __init__(self, data_path, vocab, transform):
        """
        - Training data is .json file
        - Val/Test data is .csv file 
        """
        super().__init__()

        self.data = self._load_data(data_path)

        #print(self.data)
        self.vocab = vocab
        self.transform = transform

        #load tokenizer for LLaVA
        self.nlp = spacy.load(
        'en_core_web_sm',
            exclude=[
                'attribute_ruler', 'lemmatizer', 'ner',
                'senter', 'parser', 'tagger', 'tok2vec']
        )

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.data)

    #NEED TO IMPLEMENT THIS - done
    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any, Any]:
        """
        Returns an image-utterance pair in tuple
        (img, utterance_idxs, utterance_length, raw_utterances)
        """

        # get utterance and image 
        utterance = self.data[idx]["caption"]
        img_path = self.data[idx]["image"]

        # looks like tokenization is just white space again but need to make sure this 
        # is consistent with our SayCam and LLava tokenization 
        if img_path.startswith('llava'):
            utterance_words = [token.text for token in self.nlp.tokenizer(utterance)]
        else:
            utterance_words = utterance.split()
        
        utterance_words = [SOS_TOKEN] + utterance_words + [EOS_TOKEN]
        utterance_length = len(utterance_words)

        #ensure lower case when going from token to id
        utterance_idxs = torch.tensor([self.vocab.get(
            word.lower(), UNK_TOKEN_ID) for word in utterance_words], dtype=torch.long)

        #train images need this replacement, val and test don't
        if img_path.startswith('llava'):
            img_filename = Path(img_path.replace("llava_pretrain", LLAVA_ROOT))
        elif img_path.startswith('SAYCam'):
            img_filename = Path(img_path.replace("SAYCam", SAYCAM_ROOT))
        else:
            img_filename = Path(os.path.join(LLAVA_ROOT, img_path))

        img = Image.open(img_filename).convert("RGB")
        #LLaVA images need to be resized
        img = img.resize((IMAGE_W, IMAGE_H))

        # apply transforms
        if self.transform is not None:
            img = self.transform(img)

        return img, utterance_idxs, utterance_length, [utterance]

    #NEED TO IMPLEMENT THIS - done
    def _load_data(self, data_path):
        """
        Format data based on .json (train) or .csv (val.test)
        """
        if data_path.suffix == '.json':
            return self._process_json(data_path)
        elif data_path.suffix == '.csv':
            return self._process_csv(data_path)
        else:
            raise NotImplementedError

    def _process_json(self, data_path):
        """
        - Process json dataset and format it for training
        """
        with open(data_path, "r") as f:
            data = json.load(f)
        
        # Transform to training format - convert to lowercase to match tokenization step
        formatted_data = {
            int(i): {
                "image": entry["image"],
                "caption": entry["conversations"][1]["value"].lower()
            }
            for i, entry in enumerate(data)
        }

        return formatted_data

    def _process_csv(self, data_path):
        """
        - Process csv file containg val and test data
        """
        formatted_data = {}
        
        with open(data_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for index, row in enumerate(reader):
                formatted_data[index] = {
                    "image": row["image"],
                    "caption": row["text"]
                }
        
        return formatted_data
        

class MultiModalSAYCamLLaVADataModule(MultiModalDataModule):
    """
    A data module created from baby S of the SAYCam Dataset consisting of
    image frames and the associated child-directed utterances. Also includes 
    LLaVA modified pretraining data. 
    """

    def __init__(self, args=None) -> None:
        super().__init__(args)

        #can save any args if needed here

    #no additional arguments for now
    @staticmethod 
    def add_additional_to_argparse(parser):
        return None

    @staticmethod
    def add_to_argparse(parser):
        parser = super(MultiModalSAYCamLLaVADataModule,
                       MultiModalSAYCamLLaVADataModule).add_to_argparse(parser)
        parser = MultiModalSAYCamLLaVADataModule.add_additional_to_argparse(parser)
        return parser

    #Nothing to do here 
    def prepare_data(self, *args, **kwargs) -> None:
        super().prepare_data(*args, **kwargs)

    #NEED TO IMPLEMENT THIS - done
    def read_vocab(self):
        return read_vocab(VOCAB_FILENAME)

    #Overriding this since we don't have eval datasets or the path names used in the parent class
    def setup(self, *args, **kwargs) -> None:
        print("Calling setup for Saycam + LLavA dataset!")

        # read vocab
        vocab = self.read_vocab()

        # read and create image-text data splits (train/val/test)
        self.datasets = self.create_datasets(vocab)

        # read and create eval data splits (val/test) -> they use multiple val and test methods (loss and accuracy)
        # self.eval_datasets = self.create_eval_datasets(vocab)

    #NEED TO IMPLEMENT THIS
    #Create train, val, test datasets. Dataloading and collate is handled in parent class
    def create_datasets(self, vocab):
        datasets = {}
        print("Creating datasets for Saycam + LLava mixed data...")
        
        stage_splits = [("train", TRAIN_DATA_DIR, self.transform)]

        for split, data_path, transform in stage_splits:
            dataset = MultiModalSAYCamLLaVADataset(
                data_path,
                vocab,
                transform=transform,
            )
            datasets[split] = dataset

        return datasets

    
    #Need to override dataloader creation since we don't do Labeled-S eval during training (self.eval_datasets)
    def val_dataloader(self, batch_size=None, shuffle=False, drop_last=False):
        if batch_size is None:
            batch_size = self.val_batch_size

        val_dataloader = DataLoader(
            self.datasets['val'],
            collate_fn=multiModalDataset_collate_fn,
            shuffle=shuffle,
            batch_size=batch_size,
            drop_last=drop_last,
            num_workers=self.num_workers,
            pin_memory=False,
        )

        return val_dataloader

    #Need to override dataloader creation since we don't do Labeled-S eval during training  (self.eval_datasets)
    def test_dataloader(self, batch_size=None, shuffle=False, drop_last=False):
        if batch_size is None:
            batch_size = self.val_batch_size

        test_dataloader = DataLoader(
            self.datasets['test'],
            collate_fn=multiModalDataset_collate_fn,
            shuffle=shuffle,
            batch_size=batch_size,
            drop_last=drop_last,
            num_workers=self.num_workers,
            pin_memory=False,
        )

        return test_dataloader




def tripletDataset_collate_fn(batch):
    """
    Collate function for triplet dataset that handles:
    (image, audio_text_idxs, audio_text_len, raw_audio_text, 
     touch_text_idxs, touch_text_len, raw_touch_text)
    
    Handles cases where audio_caption or touch_caption might be empty/missing.
    """
    images, audio_idxs, audio_lens, raw_audio, touch_idxs, touch_lens, raw_touch = zip(*batch)
    
    # Stack images
    images = torch.stack(images)
    
    # Pad audio text sequences (handle empty sequences)
    max_audio_len = max(audio_lens) if max(audio_lens) > 0 else 1  # Ensure at least length 1 for padding
    audio_idxs_padded = torch.zeros(len(batch), max_audio_len, dtype=torch.long)
    for i, seq in enumerate(audio_idxs):
        if len(seq) > 0:
            audio_idxs_padded[i, :len(seq)] = seq
    
    # Pad touch text sequences (handle empty sequences)
    max_touch_len = max(touch_lens) if max(touch_lens) > 0 else 1  # Ensure at least length 1 for padding
    touch_idxs_padded = torch.zeros(len(batch), max_touch_len, dtype=torch.long)
    for i, seq in enumerate(touch_idxs):
        if len(seq) > 0:
            touch_idxs_padded[i, :len(seq)] = seq
    
    audio_lens = torch.tensor(audio_lens, dtype=torch.long)
    touch_lens = torch.tensor(touch_lens, dtype=torch.long)
    
    return images, audio_idxs_padded, audio_lens, list(raw_audio), \
           touch_idxs_padded, touch_lens, list(raw_touch)


class MultiModalTripletDataset(MultiModalDataset):
    """
    Dataset that returns triplets of (image, audio_text, touch_text)
    for contrastive learning between audio and touch modalities.
    """

    def __init__(self, data_path, vocab, transform):
        """
        Expects data_path to be a .json file with both audio and touch captions.
        Format: 
        {
            "id": {
                "image": "path/to/image.jpg",
                "audio_caption": "text description from audio",
                "touch_caption": "text description from touch"
            }
        }
        """
        super().__init__()

        self.data = self._load_data(data_path)
        self.vocab = vocab
        self.transform = transform

        # Load tokenizer
        self.nlp = spacy.load(
            'en_core_web_sm',
            exclude=[
                'attribute_ruler', 'lemmatizer', 'ner',
                'senter', 'parser', 'tagger', 'tok2vec']
        )

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any, Any, Any, Any, Any]:
        """
        Returns a triplet in tuple:
        (image, audio_idxs, audio_length, raw_audio,
         touch_idxs, touch_length, raw_touch)
        
        If audio_caption or touch_caption is missing/None, returns empty tensors with length 0.
        If image/video is corrupted or missing, skips to a random other example.
        """
        try:
            # Get data
            entry = self.data[idx]

            audio_caption = entry.get("audio_caption", None)
            touch_caption = entry.get(TOUCH_CLUSTER_KEY, None)
            

            # Load and transform image
            if "image" in entry:
                img_path = entry["image"]
                if not os.path.exists(img_path):
                    raise FileNotFoundError(f"Image file not found: {img_path}")
                img = Image.open(img_path).convert("RGB")
            elif "video_path" in entry:
                video_path = entry["video_path"]
                if not os.path.exists(video_path):
                    raise FileNotFoundError(f"Video file not found: {video_path}")

                cap = cv.VideoCapture(video_path)
                total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
                center_frame = total_frames // 2

                cap.set(cv.CAP_PROP_POS_FRAMES, center_frame)
                ret, frame = cap.read()
                cap.release()

                if not ret:
                    raise RuntimeError(f"Failed to read frame from {video_path}")

                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
            else:
                raise ValueError(f"Entry {idx} has neither 'image' nor 'video_path' key")
        
            img = img.resize((IMAGE_W, IMAGE_H))
            if self.transform is not None:
                img = self.transform(img)

            # Tokenize audio caption (handle None case)
            if audio_caption and audio_caption.strip():
                
                audio_words = [token.text for token in self.nlp.tokenizer(audio_caption)]
                audio_words = [SOS_TOKEN] + audio_words + [EOS_TOKEN]
                audio_length = len(audio_words)
                audio_idxs = torch.tensor([self.vocab.get(
                    word.lower(), UNK_TOKEN_ID) for word in audio_words], dtype=torch.long)
            else:
                # Empty caption - return empty tensor with length 0
                audio_idxs = torch.tensor([], dtype=torch.long)
                audio_length = 0
                audio_caption = ""

            # Tokenize touch caption (handle None case)
            if touch_caption:
                # if type(touch_caption) == int:
                #     touch_caption = str(touch_caption)
                if type(touch_caption) == str:
                    touch_caption = touch_caption.strip()
                    touch_words = [token.text for token in self.nlp.tokenizer(touch_caption)]
                    touch_words = [SOS_TOKEN] + touch_words + [EOS_TOKEN]
                    touch_length = len(touch_words)
                    touch_idxs = torch.tensor([self.vocab.get(
                    word.lower(), UNK_TOKEN_ID) for word in touch_words], dtype=torch.long)
                elif type(touch_caption) == int:
                 
                    touch_idxs = torch.tensor([touch_caption], dtype=torch.long)
                    touch_length = 1
                else:
                    raise ValueError(f"Touch caption for entry {idx} is not a string or int: {touch_caption}")
            else:
                # Empty caption - return empty tensor with length 0
                touch_idxs = torch.tensor([], dtype=torch.long)
                touch_length = 0
                touch_caption = ""

            return img, audio_idxs, audio_length, [audio_caption], \
                touch_idxs, touch_length, [touch_caption]
        except Exception as e:
            # Skip corrupted example and try another random one
            print(f"Skipping example {idx} due to error: {e}")
            return self.__getitem__(random.randint(0, len(self)-1))

    def _load_data(self, data_path):
        """Load data from JSON file."""
        if data_path.suffix == '.json':
            return self._process_json(data_path)
        else:
            raise NotImplementedError(f"Unsupported file format: {data_path.suffix}")

    def _process_json(self, data_path):
        """
        Process json dataset and format it for triplet training.
        Expected format:
        [
            {"image": "path", "audio_caption": "text", "touch_caption": "text"},
            ...
        ]
        or 
        {
            "0": {"image": "path", "audio_caption": "text", "touch_caption": "text"},
            ...
        }
        """
        with open(data_path, "r") as f:
            data = json.load(f)
        
        # Handle list format
        if isinstance(data, list):
            return data
            formatted_data = {
                i: {
                    "image": entry["image"],
                    "audio_caption": entry.get("audio_caption", entry.get("caption", "")).lower(),
                    "touch_caption": entry.get("touch_caption", "").lower()
                }
                for i, entry in enumerate(data)
            }
        # Handle dict format
        elif isinstance(data, dict):
            formatted_data = {
                int(i): {
                    "image": entry["image"],
                    "audio_caption": entry.get("audio_caption", entry.get("caption", "")).lower(),
                    "touch_caption": entry.get("touch_caption", "").lower()
                }
                for i, entry in enumerate(data.values())
            }
        else:
            raise ValueError("Unsupported data format")

        return formatted_data


class MultiModalTripletDataModule(MultiModalDataModule):
    """
    Data module for triplet learning with audio and touch modalities.
    Requires audio and touch captions in the data files.
    """

    def __init__(self, args=None) -> None:
        super().__init__(args)

    @staticmethod
    def add_additional_to_argparse(parser):
        return None

    @staticmethod
    def add_to_argparse(parser):
        parser = super(MultiModalTripletDataModule,
                       MultiModalTripletDataModule).add_to_argparse(parser)
        parser = MultiModalTripletDataModule.add_additional_to_argparse(parser)
        return parser

    def prepare_data(self, *args, **kwargs) -> None:
        super().prepare_data(*args, **kwargs)

    def read_vocab(self):
        return read_vocab(VOCAB_FILENAME)

    def setup(self, *args, **kwargs) -> None:
        print("Setting up Triplet dataset with audio and touch modalities!")

        # read vocab
        vocab = self.read_vocab()

        # read and create image-text data splits (train/val/test)
        self.datasets = self.create_datasets(vocab)

    def create_datasets(self, vocab):
        datasets = {}
        print("Creating triplet datasets...")
        
        stage_splits = [("train", TRAIN_DATA_DIR, self.transform)]

        for split, data_path, transform in stage_splits:
            # Use TRAIN_DATA_DIR for train split, VAL_DATA_DIR for val, TEST_DATA_DIR for test
            actual_data_path = data_path
            dataset = MultiModalTripletDataset(
                actual_data_path,
                vocab,
                transform=transform,
            )
            datasets[split] = dataset

        return datasets

    def train_dataloader(self, batch_size=None, shuffle=True, drop_last=True):
        if batch_size is None:
            batch_size = self.batch_size

        train_dataloader = DataLoader(
            self.datasets['train'],
            collate_fn=tripletDataset_collate_fn,
            shuffle=shuffle,
            batch_size=batch_size,
            drop_last=drop_last,
            num_workers=self.num_workers,
            pin_memory=False,
        )

        return train_dataloader

    def val_dataloader(self, batch_size=None, shuffle=False, drop_last=False):
        if batch_size is None:
            batch_size = self.val_batch_size

        val_dataloader = DataLoader(
            self.datasets['val'],
            collate_fn=tripletDataset_collate_fn,
            shuffle=shuffle,
            batch_size=batch_size,
            drop_last=drop_last,
            num_workers=self.num_workers,
            pin_memory=False,
        )

        return val_dataloader

    def test_dataloader(self, batch_size=None, shuffle=False, drop_last=False):
        if batch_size is None:
            batch_size = self.val_batch_size

        test_dataloader = DataLoader(
            self.datasets['test'],
            collate_fn=tripletDataset_collate_fn,
            shuffle=shuffle,
            batch_size=batch_size,
            drop_last=drop_last,
            num_workers=self.num_workers,
            pin_memory=False,
        )

        return test_dataloader