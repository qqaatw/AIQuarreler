from argparse import ArgumentParser
import os
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler

from transformers import BertModel, BertTokenizer, BertConfig
from transformers import EncoderDecoderModel
from transformers.optimization import AdamW

from torchnlp.samplers import BucketBatchSampler

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from string_filters import filter_all, check_critical_words

class PTTTorchDataset(Dataset):
    def __init__(self, tsv_files, base_dir='./ptt_dataset'):
        super().__init__()
        self.dataset = {'label': [], 'input': []}

        for tsv_file in tsv_files:
            with open(os.path.join(base_dir, tsv_file), 'r', encoding='utf8') as f:
                # Fetch title
                title_list = f.readline().strip().split('\t')
                article_title_idx = title_list.index("title")
                
                for line in f:
                    content_list = line.strip().split('\t')

                    if len(content_list) != len(title_list) - 1:
                        continue
                    
                    article_title = content_list[article_title_idx]
                    for col_idx, col in enumerate(content_list):
                        # Fetch pushes
                        if col_idx != article_title_idx:
                            if len(col) == 0:
                                continue
                            if not check_critical_words(col):
                                continue

                            self.dataset['input'].append(article_title)
                            self.dataset['label'].append(col)
    
    def __len__(self):
        return len(self.dataset['label'])

    def __getitem__(self, x):
        return (filter_all(self.dataset['input'][x]), filter_all(self.dataset['label'][x]))


class PTTDataModule(LightningDataModule):
    def __init__(self, wrapped_tokenizer, hparams):
        super().__init__()
        self.wrapped_tokenizer = wrapped_tokenizer
        self.hparams = hparams
    
    def prepare_data(self, base_dir, tsv_files):
        if isinstance(tsv_files, str):
            tsv_files = [tsv_files]
        self.dataset = PTTTorchDataset(
            tsv_files=tsv_files,
            base_dir=base_dir
        )

    def setup(self):
        def collate_fn(x):
            # For dataloader.
            tokenized = self.wrapped_tokenizer([i[0] for i in x], [i[1] for i in x])
            #tokenized['input_ids'][:, -self.hparams.tokenizer_max_target_length + 1:] = tokenized['labels'][:, 1:]
            return tokenized

        def sort_key_fn(x):
            # For bucket sampler.
            return len(self.dataset[x][0])


        assert self.hparams.dataset_split_ratio >=0 and self.hparams.dataset_split_ratio <=1, \
            "dataset_split_ratio should be in [0, 1]: {}".format(self.hparams.dataset_split_ratio)

        
        num_training_samples = int(len(self.dataset) * self.hparams.dataset_split_ratio)
        self.training_dataset, self.val_dataset = random_split(self.dataset, (num_training_samples, len(self.dataset) - num_training_samples))
        
        train_sampler = BucketBatchSampler(
            SubsetRandomSampler(self.training_dataset.indices),
            batch_size=self.hparams.batch_size,
            drop_last=False,
            sort_key=sort_key_fn,
            bucket_size_multiplier=100)

        val_sampler = BucketBatchSampler(
            SubsetRandomSampler(self.val_dataset.indices),
            batch_size=self.hparams.batch_size,
            drop_last=False,
            sort_key=sort_key_fn,
            bucket_size_multiplier=100)
        
        self.train_dataloader_ = DataLoader(
            self.dataset,
            batch_sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=self.hparams.num_workers)

        self.val_dataloader_ = DataLoader(
            self.dataset,
            batch_sampler=val_sampler,
            collate_fn=collate_fn,
            num_workers=self.hparams.num_workers)

    def train_dataloader(self):
        return self.train_dataloader_

    def val_dataloader(self):
        return self.val_dataloader_


class Decoder(LightningModule):
    def __init__(
        self,
        model_name='bert-base-chinese',
        batch_size=4,
        learning_rate=1e-5,
        max_epochs=300,
        dataset_split_ratio=0.8,
        git_commit_hash=None,
        git_commit_messgae=None,
        tokenizer_padding='max_length',
        tokenizer_truncation=True,
        tokenizer_max_length=20,
        tokenizer_max_target_length=20,
        optimizer_weight_decay=0.01,
        **kwargs
        ):
        super().__init__()
        self.save_hyperparameters()
        
        self.tokenizer = BertTokenizer.from_pretrained(self.hparams.model_name, cache_dir='./.cache')
        self.wrapped_tokenizer = functools.partial(
            self.tokenizer, 
            padding=self.hparams.tokenizer_padding,
            truncation=self.hparams.tokenizer_truncation,
            max_length=self.hparams.tokenizer_max_length,
            return_tensors='pt'
        )
        self.wrapped_seq2seq_tokenizer = functools.partial(
            self.tokenizer.prepare_seq2seq_batch,
            padding=self.hparams.tokenizer_padding,
            truncation=self.hparams.tokenizer_truncation,
            max_length=self.hparams.tokenizer_max_length,
            max_target_length=self.hparams.tokenizer_max_target_length,
            return_tensors='pt'
        )

        self.base_model = EncoderDecoderModel.from_encoder_decoder_pretrained(
            encoder_pretrained_model_name_or_path=self.hparams.model_name,
            decoder_pretrained_model_name_or_path=self.hparams.model_name,
            cache_dir='./.cache',
            # Sharing parameters between encoder and decoder to reduce computational cost. 
            tie_encoder_decoder=True,
            # Pooling is no need in seq2seq.
            encoder_add_pooling_layer=False
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        return parser

    def tokenize(self, text_1, text_2=None):
        if text_2:
            return self.wrapped_seq2seq_tokenizer(src_texts=text_1, tgt_texts=text_2)
        else:
            return self.wrapped_tokenizer(text_1)

    def detokenize(self, id_tokens):
        return self.tokenizer.convert_ids_to_tokens(id_tokens)

    def configure_callbacks(self):
        checkpoint_callback = ModelCheckpoint(
            monitor='Val_CE_Loss_epoch',
            save_last=True,
            save_top_k=3,
            mode='min',
            verbose=True
        )
        earlystopping_callback = EarlyStopping(
            monitor='Val_CE_Loss_epoch',
            patience=3,
            mode='min',
            verbose=True
        )

        return [checkpoint_callback, earlystopping_callback]

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.optimizer_weight_decay,
            correct_bias=False
        )
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=0.5,
            verbose=True
        )
        
        return [optimizer], [lr_scheduler]

    def forward(self, input_ids, attention_mask, token_type_ids, decoder_input_ids):

        encoder_decoder_output =self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=True,
            return_dict=True
        )
        
        return encoder_decoder_output.logits, encoder_decoder_output
    
    def training_step(self, batch, batch_idx):
        y_input = batch.pop('labels')
        x_input = batch
        
        output_timestep_stack = []

        # Initialize decoder input with [CLS] token in position 0.
        decoder_input_ids = torch.Tensor().new_full(
            size=(y_input.shape[0], 1),
            fill_value=self.tokenizer.cls_token_id,
            dtype=y_input.dtype,
            device=self.device
        )
        for time_step, y_input_t in enumerate(y_input[0], 1):
            # TODO:
            # 1. Handle sequence when encountering sep token.
            # Reference:
            # [1] Generating Sequences With Recurrent Neural Networks 
            #     https://arxiv.org/pdf/1308.0850.pdf

            logits_output, outputs = self(decoder_input_ids=decoder_input_ids, **x_input)
            
            # Get the latest auto-regressive character.
            output_timestep_stack.append(logits_output[:, -1, :])
            
            # Mask information bahind current time step.
            decoder_input_ids = y_input.detach().clone()[:, 0:time_step + 1]

        # Stack each time-step of output to match y_input's dimensions, then apply softmax followed by log.
        log_softmax_logits = F.log_softmax(torch.stack(output_timestep_stack, dim=1), dim=-1)
        
        # Combine batch-size and time-step axises to comply input format=(N, C),
        # where N = batch-size * time-step, C = number of classes.
        loss = F.nll_loss(
            log_softmax_logits[:, :-1].reshape(-1, self.tokenizer.vocab_size),
            y_input[:,1:].reshape(-1)
        )

        self.log_dict(
            {'CE_Loss': loss},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        y_input = batch.pop('labels')
        x_input = batch
        
        output_timestep_stack = []

        # Initialize decoder input with [CLS] token in position 0.
        decoder_input_ids = torch.Tensor().new_full(
            size=(y_input.shape[0], 1),
            fill_value=self.tokenizer.cls_token_id,
            dtype=y_input.dtype,
            device=self.device
        )
        for time_step, y_input_t in enumerate(y_input[0], 1):
            # TODO:
            # 1. Handle sequence when encountering sep token.
            # Reference:
            # [1] Generating Sequences With Recurrent Neural Networks 
            #     https://arxiv.org/pdf/1308.0850.pdf

            logits_output, outputs = self(decoder_input_ids=decoder_input_ids, **x_input)
            
            # Get the latest auto-regressive character.
            output_timestep_stack.append(logits_output[:, -1, :])
            
            # Mask information bahind current time step.
            decoder_input_ids = y_input.detach().clone()[:, 0:time_step + 1]

        # Stack each time-step of output to match y_input's dimensions, then apply softmax followed by log.
        log_softmax_logits = F.log_softmax(torch.stack(output_timestep_stack, dim=1), dim=-1)
        
        # Combine batch-size and time-step axises to comply input format=(N, C),
        # where N = batch-size * time-step, C = number of classes.
        loss = F.nll_loss(
            log_softmax_logits[:, :-1].reshape(-1, self.tokenizer.vocab_size),
            y_input[:,1:].reshape(-1)
        )

        self.log_dict(
            {'Val_CE_Loss': loss},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )


if __name__ == "__main__":
    pass