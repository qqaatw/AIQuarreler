import logging
import pkg_resources
from argparse import ArgumentParser

from git import Repo
from pytorch_lightning import Trainer

from torch_model import Decoder, PTTDataModule

logging.getLogger("lightning").setLevel(logging.INFO)

# Version checking
with open("requirements.txt", 'r', encoding='utf8') as f:
    pkg_resources.require(f.read())

def train(cmd_parser):
    
    parser = Decoder.add_model_specific_args(cmd_parser)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    if args.git_record:
        repo = Repo('.')
        current_head = repo.head
        commit_hash = current_head.commit.hexsha
        commit_message = current_head.commit.message
    else:
        commit_hash = None
        commit_message = None

    decoder = Decoder(
        commit_hash = commit_hash,
        commit_message = commit_message,
        **vars(args)
    )
    
    pttdm = PTTDataModule(decoder.wrapped_seq2seq_tokenizer, decoder.hparams)
    pttdm.prepare_data(args.base_dir, args.dataset_name)
    pttdm.setup()

    trainer = Trainer.from_argparse_args(args)
    trainer.fit(decoder, pttdm)


if __name__ == "__main__":
    parser = ArgumentParser(conflict_handler='resolve')

    parser.add_argument('--dataset_name', type=str, help='TSV dataset, this can be used multiple times to add more datasets.',
        action='append', required=True)
    parser.add_argument('--base_dir', type=str, help='Directory in which TSV datasets are stored.',
        default='ptt_dataset')
    parser.add_argument('--batch_size', type=int, help='Batch size.',
        default='16')
    parser.add_argument('--dataset_split_ratio', type=float, help='Training dataset proportion. Valid range: 0~1.',
        default=0.8)
    parser.add_argument('--git_record', help='Whether to record current commit message and hash.',
        action='store_true')
    parser.add_argument('--learning_rate', type=float, help='Learning rate.',
        default=1e-5)
    parser.add_argument('--model_name', type=str, help='BERT model name.', 
        default='bert-base-chinese')
    parser.add_argument('--num_workers', type=int, help='Number of workers are used in DataLoader, \
                                                         only available in linux due to system limitations.', 
        default=0)
    
    train(parser)
