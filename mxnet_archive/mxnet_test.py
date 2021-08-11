# TODO: 1. 

import argparse
import os
import time
import logging


logging.basicConfig(
    filename=os.path.join(time.strftime('%Y%m%d-%H%M%S') + '.log'),
    level=logging.INFO
)
logging.getLogger().setLevel(logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())



import gluonnlp as nlp
import mxnet as mx
import numpy as np

from mxnet_model import Decoder, DecoderLoss
from mxnet_dataset import PTTMxnetDataset


parser = argparse.ArgumentParser(prog='mxnet_test.py')

parser.add_argument('--ckpt_dir', help='',
                    default=r'./ckpt_dir', type=str)
parser.add_argument('--resume', help='resume',
                    action='store_true')
parser.add_argument('--train', help='train',
                    action='store_true')

args = parser.parse_args()

ckpt_dir = args.ckpt_dir
nlp.utils.mkdir(ckpt_dir)



def metric_ce_fn(label, pred):
    assert label.shape[0] == pred.shape[0]
    cross_entropy = 0.0

    for batch_idx in range(label.shape[0]):
        prob = pred[batch_idx][np.arange(label[batch_idx].shape[0]), np.int64(label[batch_idx])]
        cross_entropy += (-np.log(prob + 1e-12)).sum()
    
    return (cross_entropy, len(label[0]))

#def metric_acc(label, pred):


def train():
    if args.resume:
        with open('./pre_trained/wiki_cn_cased-ddebd8f3.vocab', 'r', encoding='utf8') as f:
            bert_vocab = nlp.vocab.BERTVocab().from_json(f.read())
        decoder = mx.gluon.SymbolBlock.imports(
            os.path.join(ckpt_dir, r'epoch-symbol.json'),
            input_names=['data0', 'data1', 'data2'],
            param_file=os.path.join(ckpt_dir, r'epoch-0300.params'),
            ctx=mx.gpu())

    else:
        bert, bert_vocab = nlp.model.get_model(
            name='bert_12_768_12',
            dataset_name='wiki_cn_cased',
            pretrained=True,
            ctx=mx.gpu(),
            use_pooler=False,
            use_decoder=False,
            use_classifier=False,
            root='./pre_trained')
        
        
        decoder = Decoder(bert, 768, len(bert_vocab), 256)
        decoder.decoder.initialize(init=mx.init.Normal(0.02), ctx=mx.gpu())
        decoder.hybridize(static_alloc=False)

    tokenizer = nlp.data.BERTTokenizer(vocab=bert_vocab)
    dataset = PTTMxnetDataset(
        'hatepolitics.tsv',  
        tokenizer=tokenizer,
        column_indices=[1, 2]
    )
    non_transformed_dataset, transformed_dataset = dataset.get()
    
    for idx in range(1):
        print(non_transformed_dataset[idx])
        print(transformed_dataset[idx][0])
        input()
    #print(transformed_dataset[0].dtype)
    '''
    train_sampler = nlp.data.FixedBucketSampler(
        lengths=[item[1] for item in transformed_dataset],
        batch_size=10,
        shuffle=True)
    '''
    
    dataloader = mx.gluon.data.DataLoader(
        transformed_dataset, batch_size=50)

    loss_fn = DecoderLoss()
    loss_fn.hybridize(static_alloc=True)

    optimizer = mx.optimizer.create(
        'adam',
        learning_rate=0.00001,
        epsilon=1e-9)
    trainer = mx.gluon.Trainer(decoder.collect_params(), optimizer)
    
    params = [p for p in decoder.collect_params().values() if p.grad_req != 'null']
    
    loss_metric = mx.metric.CustomMetric(metric_ce_fn, name='CE')
    acc_metric = mx.metric.Accuracy(axis=2)

    epoch_iter = range(1, 301) if not args.resume else range(301, 601)

    for ep in epoch_iter:
        loss_metric.reset()
        acc_metric.reset()
        for batch_id, (token_ids, valid_length, segment_ids, label_token_ids, label_valid_length, label_segment_ids) in enumerate(dataloader):
            start_time = time.time()
            with mx.autograd.record():
                token_ids = token_ids.as_in_context(mx.gpu())
                valid_length = valid_length.as_in_context(mx.gpu())
                segment_ids = segment_ids.as_in_context(mx.gpu())
                label_token_ids = label_token_ids.as_in_context(mx.gpu())

                out = decoder(token_ids, segment_ids,
                                   valid_length.astype('float32'))
                ls = loss_fn(out, label_token_ids).mean()

            # Backwards computation
            ls.backward()

            # Gradient clipping
            trainer.allreduce_grads()
            nlp.utils.clip_grad_global_norm(params, 1)
            trainer.update(1)  # loss already got mean.

            # Update metric
            acc_metric.update(label_token_ids, mx.nd.softmax(out))
            loss_metric.update(label_token_ids, mx.nd.softmax(out))
            logging.info('Epoch: {} Batch: {} CE: {} ACC: {} Time: {}'.format(
                ep, batch_id, loss_metric.get_global()[1], acc_metric.get_global()[1], time.time() - start_time))
        if ep % 100 == 0:
            decoder.export(path=os.path.join(ckpt_dir, 'epoch'), epoch=ep)

def test(text):
    with open('./pre_trained/wiki_cn_cased-ddebd8f3.vocab', 'r', encoding='utf8') as f:
        vocab = nlp.vocab.BERTVocab().from_json(f.read())
    tokenizer = nlp.data.BERTTokenizer(vocab)
    transform = nlp.data.BERTSentenceTransform(
        tokenizer=tokenizer,
        max_seq_length=256,
        pair=False)
    
    transformed_text = transform(PTTMxnetDataset.filter(text))
    
    dataloader = mx.gluon.data.DataLoader(
        [transformed_text], batch_size=1)

    decoder = mx.gluon.SymbolBlock.imports(
        os.path.join(ckpt_dir, r'epoch-symbol.json'),
        input_names=['data0', 'data1', 'data2'],
        param_file=os.path.join(ckpt_dir, r'epoch-0600.params'),
        ctx=mx.gpu(1))
    
    for token_ids, valid_length, segment_ids in dataloader:
        token_ids = token_ids.as_in_context(mx.gpu(1))
        valid_length = valid_length.as_in_context(mx.gpu(1))
        segment_ids = segment_ids.as_in_context(mx.gpu(1))
        out = decoder(token_ids, segment_ids,
                      valid_length.astype('float32'))
    
    argmax = np.argmax(out.asnumpy(), axis=2).flatten().tolist()
    tokens = vocab.to_tokens(argmax)
    print(tokens)

if __name__ == "__main__":
    if args.train:
        train()
    else:
        test(r'怕了?民進黨民代撤回刪"國家統一"相關文字提案 中國台灣網5月15日訊  綜合台媒報道，民進黨籍“立委”蔡易余日前提案修改“兩岸人  民關系條例”，刪除其中“國家統一”相關文字，體現所謂“國家管轄領域僅及於台澎金  馬及其附屬島嶼”，國民黨決議不當剎車皮，讓綠營自行承擔後果。今天該案原列入台“ 立法院”報告事項，不料蔡易余主動撤案。')
