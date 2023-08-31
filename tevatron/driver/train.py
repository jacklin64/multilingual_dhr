import logging
import os
import sys

from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)

from tevatron.arguments import ModelArguments, DataArguments, ColBERTModelArguments, \
    DenseTrainingArguments as TrainingArguments
from tevatron.data import TrainDataset, TrainTASBDataset, QPCollator
from tevatron.trainer import DenseTrainer as Trainer, GCTrainer
from tevatron.datasets import HFTrainDataset, HFCorpusDataset

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments



    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)

    num_labels = 1
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        output_hidden_states=True, 
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )

    teacher_model = None
    if model_args.tct:
        if model_args.teacher_model_name_or_path is None:
            raise ValueError(
            f"when use --tct option, you should input --teacher_model_name_or_path"
        )
        # use default setting
        teacher_model_args = ColBERTModelArguments()
        teacher_model_args.model_name_or_path = model_args.teacher_model_name_or_path
        colbert_config = AutoConfig.from_pretrained(
            teacher_model_args.config_name if teacher_model_args.config_name else teacher_model_args.model_name_or_path,
            num_labels=num_labels,
            output_hidden_states=True, 
            cache_dir=teacher_model_args.cache_dir,
        )
        
        from tevatron.ColBERT.modeling import ColBERTForInference, ColBERTOutput
        from tevatron.ColBERT.modeling import ColBERTOutput as Output
        logger.info("Call model ColBERT as listwise teacher")
        teacher_model = ColBERTForInference.build(
            model_args=teacher_model_args,
            data_args=data_args,
            train_args=training_args,
            config=colbert_config,
            cache_dir=teacher_model_args.cache_dir,
        )

    if (model_args.model).lower() == 'colbert':
        from tevatron.ColBERT.modeling import ColBERT
        logger.info("Training model ColBERT")
        model = ColBERT.build(
            model_args,
            data_args,
            training_args,
            config=config,
            cache_dir=model_args.cache_dir,
        )
    elif (model_args.model).lower() == 'dhr':
        from tevatron.DHR.modeling import DHRModel
        logger.info("Training model DHR")
        model = DHRModel.build(
            model_args,
            data_args,
            training_args,
            teacher_model,
            config=config,
            cache_dir=model_args.cache_dir,
        )
    elif (model_args.model).lower() == 'dlr':
        from tevatron.DHR.modeling import DHRModel
        logger.info("Training model DLR")
        model_args.combine_cls = False
        model = DHRModel.build(
            model_args,
            data_args,
            training_args,
            teacher_model,
            config=config,
            cache_dir=model_args.cache_dir,
        )
    elif (model_args.model).lower() == 'agg':
        from tevatron.Aggretriever.modeling import DenseModel
        logger.info("Training model Dense (AGG)")
        model = DenseModel.build(
            model_args,
            data_args,
            training_args,
            config=config,
            cache_dir=model_args.cache_dir,
        )
    elif (model_args.model).lower() == 'dense':
        from tevatron.Dense.modeling import DenseModel
        logger.info("Training model Dense (CLS)")
        model = DenseModel.build(
            model_args,
            data_args,
            training_args,
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        raise ValueError('input model is not supported')


    train_dataset = HFTrainDataset(tokenizer=tokenizer, data_args=data_args,
                                   cache_dir=data_args.data_cache_dir or model_args.cache_dir)
    
    corpus_dataset = {}
    corpus_dir = data_args.corpus_dir
    # if data_args.corpus_path is None:
    for lang in data_args.lang_to_corpus_path:
        data_args.corpus_path = data_args.lang_to_corpus_path[lang]
        hf_corpus_dataset = HFCorpusDataset(tokenizer=tokenizer, data_args=data_args,
                                    cache_dir=data_args.data_cache_dir or model_args.cache_dir)
        corpus_dataset[lang] = hf_corpus_dataset.process()
    # else:
    #     hf_corpus_dataset = HFCorpusDataset(tokenizer=tokenizer, data_args=data_args,
    #                                 cache_dir=data_args.data_cache_dir or model_args.cache_dir)
    #     corpus_dataset.append(hf_corpus_dataset.process())

    ### Todo: set augument, using TASB training dataset
    # train_dataset = TrainDataset(data_args, train_dataset.process(), tokenizer)
    train_dataset = TrainTASBDataset(data_args, model_args.kd, train_dataset.process(), corpus_dataset, tokenizer)

    trainer_cls = GCTrainer if training_args.grad_cache else Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=QPCollator(
            tokenizer,
            max_p_len=data_args.p_max_len,
            max_q_len=data_args.q_max_len
        ),
    )
    train_dataset.trainer = trainer

    trainer.train()  # TODO: resume training
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
