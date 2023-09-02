from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

def list_field(default=None, metadata=None):
    return field(default_factory=lambda: default, metadata=metadata)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: Optional[str] = field(
        default="openai/whisper-large-v2",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default="ar", metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    feature_extractor_name: Optional[str] = field(
        default=None, metadata={"help": "feature extractor name or path if not the same as model_name"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    freeze_feature_encoder: bool = field(
        default=True, metadata={"help": "Whether to freeze the feature encoder layers of the model."}
    )
    freeze_encoder: bool = field(
        default=False, metadata={"help": "Whether to freeze the entire encoder of the seq2seq model."}
    )
    forced_decoder_ids: List[List[int]] = field(
        default=None,
        metadata={
            "help": (
                "A list of pairs of integers which indicates a mapping from generation indices to token indices "
                "that will be forced before sampling. For example, [[0, 123]] means the first generated token "
                "will always be a token of index 123."
            )
        },
    )
    
    suppress_tokens: List[int] = field(
        default=None, metadata={"help": "A list of tokens that will be suppressed at generation."}
    )
    model_index_name: str = field(default=None, metadata={"help": "Pretty name for the model card."})
    use_cache: Optional[bool] = field(default=False, metadata={"help": "Whether to use use_cache or not"})
    
    
@dataclass
class DataTrainingArguments:
    
    cache_dir: Optional[str] = field(
        default="./cache",
        metadata={"help": "Cache directory for huggingface data."},
    )
    
    dataset_name : str = field(
        default="fleurs",
        metadata={"help": "The name of the dataset to either load local or via the datasets library."}
    )
    load_from_local : bool = field(
        default=False,
        metadata={"help": "Whether to load the dataset locally or via the datasets library."}
    )
    load_from_disk : bool = field(
        default=False, 
        metadata={"help": "Whether Load the processed data saved in disk at dataset_dir"}
    )
    dataset_dir : str = field(
        default="google/fleurs",
        metadata={"help": "The path to the dataset directory or dataset name from huggingface datasets hub."}
    )
    dataset_config:str=field(
        default="ar",
        metadata={"help": "Dataset configuration. ar_eg for fleurs arabic, ar for common voice arabic, etc."}
    )
    dataset_duration : float = field(
        default=4.8,
        metadata={"help": "The duration of the dataset useful for few-shot learning."}
    )
    
    training_duration : float = field(
        default=1,
        metadata={"help": "The training duration for which we will train a model."}
    )
    
    max_training_duration : float = field(
        default=16,
        metadata={"help": "The maximum duration which we will train a model."}
    )
    
    text_column : Optional[str] = field(
        default="text",
        metadata={"help": "The name of the column in the datasets containing the text (as transcript)."}
    )
    audio_column : Optional[str] = field(
        default="file",
        metadata={"help": "The name of the column in the datasets containing the audio file path."}
    )
    
    sampling_rate : Optional[int] = field(
        default=16_000, 
        metadata={"help":"Audio sampling rate"}
    )
    streaming_mode : Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use streaming mode or not."}
    )
     
    train_split : Optional[str] = field(
        default="train",
        metadata={"help": "The name of the training split in the dataset."}
    )
    test_split : Optional[str] = field(
        default="test",
        metadata={"help": "The name of the training split in the dataset."}
    )
    dev_split : Optional[str] = field(
        default="dev",
        metadata={"help": "The name of the training split in the dataset."}
    )
    
    wandb_key : Optional[str] = field(
        default="ae96c42a9c89e1b5e519ad66d3e37b6db9bbe7a0",
        metadata={"help": "wandb logger"}
    )
    
    do_normalize_eval:Optional[bool] = field(
        default=False, 
        metadata={"help":"whether to normalize the text or not during evaluation"}
    )
    
    language: str = field(
        default="arabic",
        metadata={
            "help": (
                "Language for multilingual fine-tuning. This argument should be set for multilingual fine-tuning "
                "only. For English speech recognition, it should be set to `None`."
            )
        },
    )
    task: str = field(
        default="transcribe",
        metadata={"help": "Task, either `transcribe` for speech recognition or `translate` for speech translation."},
    )
    
    do_preprocessing:Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to apply processing which includes removing dicritics along with other steps. Default false"},
    )
    
    retrain_ckpt:Optional[str] = field(
        default=None,
        metadata={"help": "Path to checkpoint to retrain from"},
    )
    
    eval_ckpt:Optional[str] = field(
        default=None,
        metadata={"help": "Path to checkpoint to evaluate from"},
    )