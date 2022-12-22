#!/usr/bin/env python3
import logging
import pathlib
import re
import pandas as pd
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Union
import pickle as pkl

import datasets
import numpy as np
import torch
import torch.nn as nn
from packaging import version

import librosa
from scipy.io import wavfile
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    is_apex_available,
    trainer_utils,
)

def main():
    sample_to_feats_dict_layer_1 = {}
    sample_to_feats_dict_layer_12 = {}
    sample_to_feats_dict_layer_24 = {}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_type="ft"
    lang="sv"
    k=4
    save_asr_outputs=False
    
    if lang=="sv":
        path_to_csv="/scratch/work/getmany1/swedish_df.csv"
        df = pd.read_csv(path_to_csv, encoding='utf-8')
        if model_type=="ft":
            checkpoint_list=['5548','5567','5339','5491']
            model_path="/scratch/elec/puhe/p/getmany1/wav2vec2-large-voxrex-KBLab_vocab-digitala_SLT2022"
        elif model_type=="pt":
            model_path="KBLab/wav2vec2-large-voxrex"
        processor = Wav2Vec2Processor.from_pretrained("KBLab/wav2vec2-large-voxrex-swedish",cache_dir="/scratch/elec/puhe/p/getmany1/cache")
    elif lang=="fi":
        path_to_csv="/scratch/work/getmany1/finnish_df.csv"
        df = pd.read_csv(path_to_csv, encoding='utf-8')
        if model_type=="ft":
            checkpoint_list=['7920','7900','7840','8020']
            model_path="/scratch/elec/puhe/p/getmany1/wav2vec2-large-14.2k-fi-digitala_SLT2022"
        elif model_type=="pt":
            model_path="/scratch/work/getmany1/wav2vec/wav2vec2_large_14.2k_fi_0902022"
        processor = Wav2Vec2Processor.from_pretrained("/scratch/work/getmany1/wav2vec/wav2vec2_large_14.2k_fi_0902022",cache_dir="/scratch/elec/puhe/p/getmany1/cache")
    wer_metric = datasets.load_metric("wer")
    cer_metric = datasets.load_metric("cer")

    if model_type=="ft":
        for i in range(k):
            print(f"Fold {i}")
            df_temp = df[df.split==i]
            model=Wav2Vec2ForCTC.from_pretrained(f"{model_path}_fold_{str(i)}/checkpoint-{checkpoint_list[i]}", cache_dir="/scratch/elec/puhe/p/getmany1/cache", pad_token_id=processor.tokenizer.pad_token_id, vocab_size=len(processor.tokenizer)).to(device)

            for idx, row in df_temp.iterrows():
                sample_id = row['sample']
                speech, _ = librosa.load(row["recording_path"], sr=16000)
                input_values = processor(speech, sampling_rate=16000, return_tensors="pt", padding="longest").input_values
                with torch.no_grad():
                    outputs = model(input_values.to(device), output_hidden_states=True)
                    # logits = model(input_values.to(device)).logits
                logits = outputs.logits

                feats_layer_1 = outputs.hidden_states[-24].cpu().numpy()[0] # embeddings of the first hidden layer of shape (seq_len, 1024)
                feats_layer_12 = outputs.hidden_states[-12].cpu().numpy()[0] # embeddings of the middle hidden layer of shape (seq_len, 1024)
                feats_layer_24 = outputs.hidden_states[-1].cpu().numpy()[0] # embeddings of the last hidden layer of shape (seq_len, 1024)
    
                sample_to_feats_dict_layer_1[sample_id] = feats_layer_1
                sample_to_feats_dict_layer_12[sample_id] = feats_layer_12
                sample_to_feats_dict_layer_24[sample_id] = feats_layer_24

                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = processor.batch_decode(predicted_ids)[0]
                df.loc[df[df.recording_path==row.recording_path].index.item(), "ASR_transcript"] = transcription
        if save_asr_outputs:
            df.to_csv(f"/scratch/work/getmany1/wav2vec/slt_2022/{lang}/swedish_df_with_asr_transcripts.csv",encoding='utf-8',index=False)

    elif model_type=="pt":
        model=Wav2Vec2ForCTC.from_pretrained(model_path, cache_dir="/scratch/elec/puhe/p/getmany1/cache").to(device)
        for idx, row in df.iterrows():
            sample_id = row['sample']
            speech, _ = librosa.load(row["recording_path"], sr=16000)
            input_values = processor(speech, sampling_rate=16000, return_tensors="pt", padding="longest").input_values
            with torch.no_grad():
                outputs = model(input_values.to(device), output_hidden_states=True)

            feats_layer_1 = outputs.hidden_states[-24].cpu().numpy()[0] # embeddings of the first hidden layer of shape (seq_len, 1024)
            feats_layer_12 = outputs.hidden_states[-12].cpu().numpy()[0] # embeddings of the middle hidden layer of shape (seq_len, 1024)
            feats_layer_24 = outputs.hidden_states[-1].cpu().numpy()[0] # embeddings of the last hidden layer of shape (seq_len, 1024)

            sample_to_feats_dict_layer_1[sample_id] = feats_layer_1
            sample_to_feats_dict_layer_12[sample_id] = feats_layer_12
            sample_to_feats_dict_layer_24[sample_id] = feats_layer_24

    pkl.dump(sample_to_feats_dict_layer_1, open(f"/scratch/work/getmany1/wav2vec/slt_2022/{lang}/{model_type}/w2v2_digitala_feats_dict_{lang}_{model_type}_layer_1.pkl", "wb"))
    pkl.dump(sample_to_feats_dict_layer_12, open(f"/scratch/work/getmany1/wav2vec/slt_2022/{lang}/{model_type}/w2v2_digitala_feats_dict_{lang}_{model_type}_layer_12.pkl", "wb"))
    pkl.dump(sample_to_feats_dict_layer_24, open(f"/scratch/work/getmany1/wav2vec/slt_2022/{lang}/{model_type}/w2v2_digitala_feats_dict_{lang}_{model_type}_layer_24.pkl", "wb"))

if __name__ == "__main__":
    main()
