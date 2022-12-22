from datasets import load_dataset, Dataset, load_metric, Audio
import torch
from transformers import pipeline, AutoModelForAudioClassification,AutoFeatureExtractor, AutoTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, AutoModelForAudioClassification, TrainingArguments, Trainer
import soundfile as sf
import librosa
import numpy as np
from scipy.stats import spearmanr
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, recall_score, accuracy_score, f1_score#, spearmanr
from pickle import load, dump

def true_round_and_convert_to_label(x):
    import decimal
    return int(decimal.Decimal(str(x)).quantize(decimal.Decimal("1"), rounding=decimal.ROUND_HALF_UP))-2

def true_round(x):
    import decimal
    return int(decimal.Decimal(str(x)).quantize(decimal.Decimal("1"), rounding=decimal.ROUND_HALF_UP))

def process(df, round_labels=True):
    if round_labels:
        df['label'] = df['label'].map(np.vectorize(true_round))
    if any(x in df.columns for x in ["Unnamed: 0", "identifier","n_sentences"]):
        df.drop(columns=["Unnamed: 0", "identifier","n_sentences"], inplace=True)
    df.fillna(0.0, inplace=True)
    # drop sparse features
    df = df.loc[:, (df != 0).any(axis=0)]
    df = df[df['label'].apply(threshold)]
    df['label'] = df['label'].map(lambda a: int(a - 2))
    return df

def compute_metrics(pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(pred.predictions, axis=1)
    accuracy = accuracy_score(pred.label_ids, predictions)
    uar = recall_score(pred.label_ids, predictions, average='macro')
    f1 = f1_score(pred.label_ids, predictions, average='macro')
    return {"accuracy": accuracy, "uar": uar, "f1": f1}

def prepare_example(example): 
    if '.FI0' in example["file"]:
        example["speech"], example["sampling_rate"] = sf.read(example["file"], channels=1, samplerate=16000, format='RAW', subtype='PCM_16')
    else:
        example["audio"], example["sampling_rate"] = librosa.load(example["file"], sr=16000)
    example["duration_in_seconds"] = len(example["audio"]) / 16000
    return example

def preprocess_function(examples):
    audio_arrays = examples["audio"]
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=feature_extractor.sampling_rate
    )
    return inputs

def map_to_array(example):
    speech, _ = librosa.load(example["file"], sr=16000, mono=True)
    example["speech"] = speech
    return example

def map_to_pred(batch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_values = processor(batch["audio"], sampling_rate=16000, return_tensors="pt", padding="longest").input_values
    with torch.no_grad():
        logits = model(input_values.to(device)).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    batch["predictions"] = predicted_ids
    return batch

STAGE="TRAIN"
# model_type="RAW_pt"
k = 4
target_sr = 16000

if STAGE=="TRAIN":
    freeze_feature_extractor=True
    batch_size = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    f1_metric = load_metric("f1")
    accuracy_metric = load_metric("accuracy")
    spearmanr_metric = load_metric("spearmanr")
    lang="sv"
    model_type="ASR_ft"
    if lang=="fi":
        num_labels = 6
        threshold = lambda x: x >= 2.0
        df = pd.read_csv("/scratch/work/getmany1/finnish_df.csv", usecols=['recording_path','cefr_mean','split'])
        if model_type=="ASR_ft":
            checkpoint_list=['7920','7900','7840','8020']
            model_path="/scratch/elec/puhe/p/getmany1/wav2vec2-large-14.2k-fi-digitala_SLT2022"
        elif model_type=="RAW_pt":
            model_path="/scratch/work/getmany1/wav2vec/wav2vec2_large_14.2k_fi_0902022"
        processor = Wav2Vec2Processor.from_pretrained("/scratch/work/getmany1/wav2vec/wav2vec2_large_14.2k_fi_0902022",cache_dir="/scratch/elec/puhe/p/getmany1/cache")
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("/scratch/work/getmany1/wav2vec/wav2vec2_large_14.2k_fi_0902022",cache_dir="/scratch/elec/puhe/p/getmany1/cache")
    elif lang=="sv":
        num_labels = 4
        threshold = lambda x: 2.0 <= x <= 5.0
        df = pd.read_csv("/scratch/work/getmany1/swedish_df.csv", usecols=['recording_path','cefr_mean','split'])
        if model_type=="ASR_ft":
            checkpoint_list=['5548','5567','5339','5491']
            model_path="/scratch/elec/puhe/p/getmany1/wav2vec2-large-voxrex-KBLab_vocab-digitala_SLT2022"
        elif model_type=="RAW_pt":
            model_path="KBLab/wav2vec2-large-voxrex"
        processor = Wav2Vec2Processor.from_pretrained("KBLab/wav2vec2-large-voxrex-swedish",cache_dir="/scratch/elec/puhe/p/getmany1/cache")
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("KBLab/wav2vec2-large-voxrex-swedish",cache_dir="/scratch/elec/puhe/p/getmany1/cache")

    df.rename(columns={'cefr_mean': 'label'}, inplace=True)
    df.rename(columns={'recording_path': 'file'}, inplace=True)
    df=process(df)

    k = 4
    for i in range(k):
        print(f"Fold {i}")
        if model_type=="ASR_ft":
            model=AutoModelForAudioClassification.from_pretrained(f"{model_path}_fold_{str(i)}/checkpoint-{checkpoint_list[i]}", cache_dir="/scratch/elec/puhe/p/getmany1/cache", num_labels=num_labels).to(device)
        elif model_type=="RAW_pt":
            model=AutoModelForAudioClassification.from_pretrained(model_path, cache_dir="/scratch/elec/puhe/p/getmany1/cache",num_labels=num_labels).to(device)

        df_train_temp = df[df.split!=i]
        df_train_temp.drop(columns=["split"], inplace=True)
        df_dev_temp = df[df.split==i]
        df_dev_temp.drop(columns=["split"], inplace=True)
        train_dataset = Dataset.from_pandas(df_train_temp)
        dev_dataset = Dataset.from_pandas(df_dev_temp)

        train_dataset = train_dataset.map(prepare_example, remove_columns=['file'])
        dev_dataset = dev_dataset.map(prepare_example, remove_columns=['file'])
        train_dataset = train_dataset.map(preprocess_function, batched=True, batch_size=1)
        dev_dataset = dev_dataset.map(preprocess_function, batched=True, batch_size=1)

        output_dir=f"/scratch/elec/puhe/p/getmany1/wav2vec2_large_for_classification_digitala_{lang}_{model_type}_frozen_FE_fold_{str(i)}"

        args = TrainingArguments(
            output_dir,
            evaluation_strategy = "epoch",
            save_strategy = "epoch",
            logging_strategy="epoch",
            learning_rate=3e-5,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=1,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=10,
            warmup_ratio=0.1,
            load_best_model_at_end=False,
            metric_for_best_model="f1",
            push_to_hub=False,
            gradient_checkpointing=True,
            save_total_limit=1
        )

        if freeze_feature_extractor:
            model.freeze_feature_extractor()

        trainer = Trainer(
            model,
            args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            tokenizer=feature_extractor,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        print(f"Fold {i} finished.")

elif STAGE=="EVAL":
    k=4
    batch_size = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for lang in ["sv","fi"]:
        if lang=="sv":
            threshold = lambda x: 2.0 <= x <= 5.0
            df = pd.read_csv("/scratch/work/getmany1/swedish_df.csv", usecols=['recording_path','cefr_mean','split'])
            checkpoint_list=['5730','5710','5500','5660']
            processor = Wav2Vec2Processor.from_pretrained("KBLab/wav2vec2-large-voxrex-swedish",cache_dir="/scratch/elec/puhe/p/getmany1/cache")
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("KBLab/wav2vec2-large-voxrex-swedish",cache_dir="/scratch/elec/puhe/p/getmany1/cache")
        elif lang=="fi":
            threshold = lambda x: x >= 2.0
            df = pd.read_csv("/scratch/work/getmany1/finnish_df.csv", usecols=['recording_path','cefr_mean','split'])
            checkpoint_list=['7910','7890','7840','8020']
            processor = Wav2Vec2Processor.from_pretrained("/scratch/work/getmany1/wav2vec/wav2vec2_large_14.2k_fi_0902022",cache_dir="/scratch/elec/puhe/p/getmany1/cache")
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("/scratch/work/getmany1/wav2vec/wav2vec2_large_14.2k_fi_0902022",cache_dir="/scratch/elec/puhe/p/getmany1/cache")
        df.rename(columns={'cefr_mean': 'label'}, inplace=True)
        df.rename(columns={'recording_path': 'file'}, inplace=True)
        df=process(df)        
        
        for model_type in ["RAW_pt","ASR_ft"]:
            for fe in ["","_frozen_FE"]:
                y_test_full = np.empty(0)
                y_pred_full = np.empty(0)
                for i in range(k):
                    df_test_temp = df[df.split==i]
                    df_test_temp.drop(columns=["split"], inplace=True)
                    test_dataset = Dataset.from_pandas(df_test_temp)
                    test_dataset = test_dataset.map(prepare_example, remove_columns=['file'])
                    test_dataset = test_dataset.map(preprocess_function, batched=True, batch_size=1)

                    model=AutoModelForAudioClassification.from_pretrained(f"/scratch/elec/puhe/p/getmany1/wav2vec2_large_for_classification_digitala_{lang}_{model_type}{fe}_fold_{str(i)}/checkpoint-{checkpoint_list[i]}",cache_dir="/scratch/elec/puhe/p/getmany1/cache").to(device)
                    result = test_dataset.map(map_to_pred)

                    y_test_full = np.append(y_test_full,np.array(result['label']))
                    y_pred_full = np.append(y_pred_full,np.array(result['predictions']))

                print(f"Results. {lang}. {model_type}. {fe}")
                print(classification_report(y_test_full, y_pred_full, digits=4))

                dump(y_test_full, open(f"/scratch/work/getmany1/wav2vec/slt_2022/{lang}/labels.pkl", 'wb'))
                dump(y_pred_full, open(f"/scratch/work/getmany1/wav2vec/slt_2022/{lang}/class_ft_w2v2_outputs/predictions_{model_type}{fe}.pkl", 'wb'))
