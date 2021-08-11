import mxnet as mx
import mxnet.gluon.nn as nn

class Decoder(mx.gluon.HybridBlock):
    def __init__(self, bert, units, vocab_size, max_seq_length, prefix = None):
        """ Construct a decoder for the masked language model task """
        super(Decoder, self).__init__()
        with self.name_scope():
            self.bert = bert
            self.max_seq_length = max_seq_length
            self.decoder = nn.HybridSequential(prefix=prefix)
            self.decoder.add(nn.Dense(units, flatten=False))
            self.decoder.add(nn.GELU())
            self.decoder.add(nn.LayerNorm(in_channels=units, epsilon=1e-12))
            self.decoder.add(nn.Dense(vocab_size, flatten=False))
        #assert self.decoder[3].weight == list(embed.collect_params().values())[0], \
        #    'The weights of word embedding are not tied with those of decoder'
    
    def hybrid_forward(self, F, token_ids, segment_ids, valid_length):
        bert = self.bert(token_ids, segment_ids, valid_length)
        #only_repr = F.slice(bert, begin=(None, 1, None), end=(None, self.max_seq_length, None))
        return self.decoder(bert)

class DecoderLoss(mx.gluon.HybridBlock):
    def __init__(self):
        super(DecoderLoss, self).__init__()
        self.loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()
    
    def hybrid_forward(self, F, pred, label, *args, **kwargs):
        return self.loss(pred, label)
