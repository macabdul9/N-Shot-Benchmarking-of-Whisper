#!/usr/bin/env python3
import os
import sys
import logging
import jiwer
from copy import deepcopy
import pandas as pd
## Read the audio files in the data
from datasets import Audio
import transformers
from transformers import (
    HfArgumentParser,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
    pipeline,
    EarlyStoppingCallback
)

from utils import (
    data_loader,
    preprocessor,
    sampler,
    DataCollatorSpeechSeq2SeqWithPadding,
    write_txt_file,
    save_json,
    lmap
    
)
from config import (
    ModelArguments,
    DataTrainingArguments,
)

import datasets 
from datasets import load_from_disk


from transformers.trainer_utils import get_last_checkpoint, is_main_process

import torch
import gc
import json 

logger = logging.getLogger(__name__)
# os.environ['TRANSFORMERS_CACHE'] = "./cache"
os.environ["WANDB_DISABLED"] = "true"

def handle_metrics(split, metrics, output_dir):
    """
    Log and save metrics
    Args:
    - split: one of train, val, test
    - metrics: metrics dict
    - output_dir: where to save the metrics
    """

    logger.info(f"***** {split} metrics *****")
    for key in sorted(metrics.keys()):
        logger.info(f"  {key} = {metrics[key]}")
    save_json(metrics, os.path.join(output_dir, f"{split}_results.json"))


def main():
    
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    
    # 1. parser
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        
    
    print("Starting")

    # 5. root dirctories = output_dir/dataset_name
    root_dir = os.path.join(training_args.output_dir, data_args.dataset_name)
    os.makedirs(root_dir, exist_ok=True)


    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Loggin -> from ASRs/train.py 
    ## Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    ## Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Training/Data parameters %s", data_args)

    ## Set seed before initializing model.
    set_seed(training_args.seed)
    
    
    # 2. create tokenizer and prerpcoessor which is going to be same for all the models
    tokenizer = WhisperTokenizer.from_pretrained(
        model_args.model_name_or_path, 
        cache_dir=data_args.cache_dir,
        language="arabic", 
        task="transcribe"
    )
        
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_args.model_name_or_path, cache_dir=data_args.cache_dir)
    
    # processor and data collator
    processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    
    # 6.4 create data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    
    # apply feature extractor and tokenizer to dataset
    def prepare_dataset(batch):
       
        # load and resample audio data from 48 to 16kHz
        audio = batch[data_args.audio_column] # after resampling, audio is a dict with keys "array" and "sampling_rate"

        # compute log-Mel input features from input audio array 
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        
        # encode target text to label ids 
        batch["labels"] = tokenizer(batch[data_args.text_column]).input_ids
        return batch  
    
    # # metrics function to compute metrics 
    # wer_metric = evaluate.load("wer", cache_dir=data_args.cache_dir)
    # cer_metric = evaluate.load("cer", cache_dir=data_args.cache_dir)
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * jiwer.wer(label_str, pred_str) #wer_metric.compute(predictions=pred_str, references=label_str)
        cer = 100 * jiwer.cer(label_str, pred_str) #cer_metric.compute(predictions=pred_str, references=label_str)
        

        return {"wer": wer, "cer": cer}
    
    
    
    # See if processed data is available
    if data_args.load_from_disk:
        print("Loading the processed data from disk")
        dataset = datasets.load_from_disk(data_args.dataset_dir)
    
    else: 
        # 1. load the appropriate dataset
        print("Loading raw data from hub or local!")
        dataset = data_loader(load_from_local=data_args.load_from_local, dataset_dir=data_args.dataset_dir, cache_dir=data_args.cache_dir) 
    
    # sample train dataset  sample only if training_duration < data_duration
    if data_args.training_duration < data_args.dataset_duration:
        print(f'Sampling data for {data_args.training_duration} hrs')
        dataset['train'] = sampler(
            dataset=dataset['train'], 
            dataset_duration=data_args.dataset_duration, 
            training_duration=data_args.training_duration
        )
    
    ## if processed data is saved as output_dir/data then load it -> this is applicable only for test and dev set
    if not data_args.load_from_disk:
        data_path = os.path.join(root_dir, "data")
        if os.path.exists(data_path):
            # load test and validation set and process train set
            print(F"Found preprocessed test and dev splits at: {data_path}")
            if data_args.do_preprocessing:
                dataset[data_args.train_split] = dataset['train'].map(lambda x: {data_args.text_column: preprocessor(x[data_args.text_column])}).filter(lambda x: len(x[data_args.text_column]) > 1).cast_column(data_args.audio_column, Audio(sampling_rate=data_args.sampling_rate))
            else:
                
                dataset[data_args.train_split] = dataset['train'].filter(lambda x: len(x[data_args.text_column]) > 1).cast_column(data_args.audio_column, Audio(sampling_rate=data_args.sampling_rate))
            
            dataset[data_args.train_split] = dataset['train'].map(prepare_dataset, remove_columns=dataset.column_names["train"])
            
            # load preprocessed test and dev sets here
            dataset[data_args.test_split] = load_from_disk(os.path.join(data_path, "test.hf"))
            dataset[data_args.dev_split] = load_from_disk(os.path.join(data_path, "dev.hf"))
            
        else:
            print(f"No processed data found!")
        
            # 3. Data Processor 
            ## 3.1 Apply text normalizer (.map) 3.2 Remove examples with empty text (.filter) after the normalization 3.3 Read  audio file 
            for split in dataset:
                
                if data_args.do_preprocessing:
                    dataset[split] = dataset[split].map(lambda x: {data_args.text_column: preprocessor(x[data_args.text_column])}).filter(lambda x: len(x[data_args.text_column]) > 1).cast_column(data_args.audio_column, Audio(sampling_rate=data_args.sampling_rate))
                else:
                    dataset[split] = dataset[split].filter(lambda x: len(x[data_args.text_column]) > 1).cast_column(data_args.audio_column, Audio(sampling_rate=data_args.sampling_rate))

            # 4. Apply tokenizer and feature extractor on text and audio array respectively 
            dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"])
            
            # save the test and dev/validation set
            os.makedirs(data_path, exist_ok=True)
            
            print(f"Saving the preprocessed test and dev splits at: {data_path}")
            dataset["test"].save_to_disk(os.path.join(data_path, "test.hf"))
            dataset["dev"].save_to_disk(os.path.join(data_path, "dev.hf"))
        
    
    # to aggregate the resule in single csv file
    few_shot_metrics = {}
    few_shot_metrics_path = os.path.join(root_dir, data_args.dataset_name+"_all_results.json")
    # see if the file exist
    if os.path.exists(few_shot_metrics_path):
        with open(few_shot_metrics_path,"r") as f:
            few_shot_metrics = json.load(f)

    # 6.3 create directories for saving results, models and logging corresponding to each training setting
    output_dir = os.path.join(root_dir, str(data_args.training_duration)) # this is where training metrics will be saved
    # ckpt_dir = os.path.join(output_dir, 'ckpt') # # this is where training checkpoints will be saved
    os.makedirs(output_dir, exist_ok=True)
    # os.makedirs(ckpt_dir, exist_ok=True)
    
    # 6.5 load pretrained to finetune or evaluate
    if data_args.retrain_ckpt or data_args.eval_ckpt:
        print(f"Loading the pretrained model from {data_args.retrain_ckpt} or {data_args.eval_ckpt} for evaluation or retraining!")
        model = WhisperForConditionalGeneration.from_pretrained(data_args.retrain_ckpt if data_args.retrain_ckpt else data_args.eval_ckpt, cache_dir=data_args.cache_dir)
    else:
        print("Loading the pretrained model from hub or local!")
        model = WhisperForConditionalGeneration.from_pretrained(model_args.model_name_or_path, cache_dir=data_args.cache_dir)
        
    model.config.use_cache=False
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    

    # 6.6 create directories for saving results, models and logging corresponding to each training setting
    output_dir = os.path.join(root_dir, str(data_args.training_duration))
    # ckpt_dir = os.path.join(output_dir, 'ckpt')
    
    # if it exists either remove the director or 
    os.makedirs(output_dir, exist_ok=True)
    # os.makedirs(ckpt_dir, exist_ok=True)
    
    
    training_args.output_dir = output_dir
    
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[data_args.train_split],
        eval_dataset=dataset[data_args.dev_split],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.01)]
    )
    
    
    # # wandb configuration only on main process 
    # if is_main_process(training_args.local_rank):
    #     wandb.login(key=data_args.wandb_key)
    #     wandb.init(
    #         project="whisper-finetuning",
    #         group=data_args.dataset_name,
    #         name=str(data_args.training_duration),
    #         tags=str(data_args.training_duration),
    #     )
    
    
    
    all_metrics = {}
    
    # Training
    if training_args.do_train:
        
        logger.info("*** Training ***")
                
        train_result = trainer.train()
        
        metrics = train_result.metrics
        
        metrics.update({"total_duration":data_args.dataset_duration})
        metrics.update({"training_duration":data_args.training_duration})
        metrics.update({"training_examples":len(dataset[data_args.train_split])})

        
        print(f"Saving model at {output_dir}!")
        trainer.save_model()  # this also saves the tokenizer
            
            
        handle_metrics("train", metrics, output_dir)
        all_metrics.update(metrics)

        # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
        # trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))
        trainer.save_state()
        
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)


    # # Evaluation -> This is redundant when evaluating along during prediction -> Uncomment it if you don't want to do the predicitons
    # if training_args.do_eval:
        
    #     logger.info("*** Evaluating ***")
        
    #     metrics = trainer.evaluate(metric_key_prefix="eval")

    #     handle_metrics("dev", metrics, output_dir)
    #     all_metrics.update(metrics)
            
    if training_args.do_predict:
        
        logger.info("*** Predicting ***")
        
        # force model to transcribe arabic transcirpt only -> tested for accurate prediction -> results/fleurs-1hr and test.py 
        trainer.model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="ar", task=data_args.task)
    
        test_output = trainer.predict(test_dataset=dataset['test'], metric_key_prefix="test")
        eval_output = trainer.predict(test_dataset=dataset['dev'], metric_key_prefix="eval")
        
        test_metrics = test_output.metrics
        eval_metrics = eval_output.metrics
        
                
        test_metrics["test_loss"] = round(test_metrics["test_loss"], 4)
        eval_metrics["eval_loss"] = round(eval_metrics["eval_loss"], 4)
        
        handle_metrics("test", test_metrics, output_dir)
        handle_metrics("eval", eval_metrics, output_dir)
        
        all_metrics.update(test_metrics)
        all_metrics.update(eval_metrics)

        if training_args.predict_with_generate:
            
            test_preds = tokenizer.batch_decode(test_output.predictions, skip_special_tokens=True)
    
            eval_preds = tokenizer.batch_decode(eval_output.predictions, skip_special_tokens=True)
            
            test_preds = lmap(str.strip, test_preds)
            eval_preds = lmap(str.strip, eval_preds)
            write_txt_file(test_preds, os.path.join(output_dir, "test_predictions.txt"))
            write_txt_file(eval_preds, os.path.join(output_dir, "eval_predictions.txt"))

    
    save_json(all_metrics, os.path.join(output_dir, "results.json"))

    # add the key metrics to few-shot results 
    few_shot_metrics.update({
        str(data_args.training_duration)+"/"+str(data_args.dataset_duration):{k: round(v, 2) for k, v in all_metrics.items() if k.endswith(('loss', 'wer', 'cer'))}
    })
    
    # update the combined results and save 
    save_json(few_shot_metrics, few_shot_metrics_path)
    

    torch.cuda.empty_cache()
    gc.collect()

    
if __name__ == '__main__':
    
    main()
    