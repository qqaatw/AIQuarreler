import os
import re

from argparse import ArgumentParser
from glob import glob

import torch
from torch.nn import functional as F

from torch_model import Decoder
from string_filters import filter_all

class PredefinedInference:
    def __init__(self, ckpt_path):
        self.decoder = Decoder.load_from_checkpoint(ckpt_path)
        self.decoder.freeze()
        self.generate_conf = [
            {}, # Greedy Search
            {"num_beams": 2, "num_return_sequences": 1},
            {"num_beams": 3, "num_return_sequences": 1},
            {"num_beams": 4, "num_return_sequences": 1},
            {"do_sample": True, "top_p": 0.92},
            {"do_sample": True, "top_k": 5}
        ]
    
    def generate(self, text, conf_no=None):
        if conf_no is None:
            results = []
            for conf in self.generate_conf:
                _, generated = transformers_generate(self.decoder, filter_all(text.strip()), **conf)
                results.append(generated[0])
            return results, None
        else:
            _, generated = transformers_generate(self.decoder, filter_all(text.strip()), **self.generate_conf[conf_no])
            return generated[0], self.generate_conf[conf_no]


def transformers_generate(decoder, filtered_text, **generate_kwargs):
    # 1. Given a decoder with trained encoder outputs.
    # 2. Using single time-step input_id as decoder input.
    # 3. After decoder output, concatenate t-1 output to decoder input_ids.
    # 4. Repeating step 2~3 until the eos token is reached.

    inputs = decoder.tokenize(filtered_text)
    
    outputs = decoder.base_model.generate(
        **inputs, 
        bos_token_id=decoder.tokenizer.cls_token_id,
        pad_token_id=decoder.tokenizer.pad_token_id,
        eos_token_id=decoder.tokenizer.sep_token_id,
        bad_words_ids=[decoder.tokenizer(bad_word).input_ids for bad_word in ["推", "樓下", "[UNK]", "被肛"]],
        use_cache=True,
        length_penalty=1,
        # Important
        no_repeat_ngram_size=2,
        **generate_kwargs
    )

    return decoder.detokenize(outputs[0].tolist()), \
        decoder.tokenizer.batch_decode(
            outputs.tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

def my_generate(decoder, filtered_text): # deprecated
    # 1. Given a decoder with trained encoder outputs.
    # 2. Using full input_ids as decoder input.
    # 3. After decoder output, mask info behind current time-step by 0(pad token).
    # 4. Repeating step 2~3 until the eos token is reached.

    inputs = decoder.tokenize(filtered_text)
   
    decoder_input_ids = torch.Tensor().new_full(
        (1, 20),
        fill_value=decoder.tokenizer.pad_token_id,
        dtype=torch.int64
    )
    output_timestep_stack = []
    for time_step in range(decoder.hparams.tokenizer_max_length):
        logits_output = decoder(decoder_input_ids=decoder_input_ids, **inputs)
        softmax_logits_output = F.softmax(logits_output, dim=-1)
        
        
        current_prob_rank = torch.argsort(softmax_logits_output[0, time_step], descending=True)
        print('Prob rank', decoder.detokenize(current_prob_rank)[:10])
        
        if time_step != 0:
            if current_prob_rank[0] == decoder.tokenizer.sep_token_id:
                break
            for prob in current_prob_rank:
                # If [CLS], [PAD], [UNK], 推, then pass to next prob.
                skip_ids = [
                    decoder.tokenizer.cls_token_id,
                    decoder.tokenizer.pad_token_id,
                    2972,
                    100
                ]
                if prob not in skip_ids:
                    output_timestep_stack.append(prob)
                    decoder_input_ids[:, time_step] = prob
                    break
        else:
            decoder_input_ids = torch.argmax(softmax_logits_output, dim=-1, keepdim=False)
        decoder_input_ids[:, time_step +1:] = 0
        #decoder_input_ids = decoder_input_ids[:, :time_step+1]
    detokenized_text = decoder.detokenize(torch.stack(output_timestep_stack, dim=0).tolist())
    return detokenized_text, decoder.tokenizer.batch_decode(outputs.tolist(), skip_special_tokens=True)

def main(ckpt_path_1, ckpt_path_2, text, config_no):
    print(f'Using ckpt: {ckpt_path_1}, {ckpt_path_2}, Config NO: {config_no}')

    if text:
        filtered_texts = [filter_all(text.strip())]
    else:
        filtered_texts = []
        with open('testing_text.txt', 'r', encoding='utf8') as f:
            for line in f:
                if line[0] in ['#', '\n']: continue 
                filtered_texts.append(filter_all(line.strip()))

    outputs = []
    if ckpt_path_2:
        inference_1 = PredefinedInference(ckpt_path_1)
        inference_2 = PredefinedInference(ckpt_path_2)
        for filtered_text in filtered_texts:
            detokenized_text_1 = inference_1.generate(filtered_text, conf_no=config_no)
            detokenized_text_2 = inference_2.generate(filtered_text, conf_no=config_no)
            outputs.append(f"Input:{filtered_text}, Output1:{detokenized_text_1}, Output2:{detokenized_text_2}")
    else:
        inference = PredefinedInference(ckpt_path_1)
        for filtered_text in filtered_texts:
            detokenized_text_1 = inference.generate(filtered_text, conf_no=config_no)
            outputs.append(f"Input:{filtered_text}, Output1:{detokenized_text_1}")
    print('\n'.join(outputs))

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('-t', '--text', type=str, help="Text to predict. If not given, use the texts in testing_text.txt",
        default=None)
    parser.add_argument('--path2', type=str, help="Second ckpt path if a comparison among two ckpts is needed.",
        default=None)
    parser.add_argument('--config_no', type=int, help="Config number. See README.md for details.",
        default=2)
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument('--path', type=str, help="First ckpt path.",
        default=None)
    group.add_argument('--latest', help="Use the latest checkpoint.",
        action="store_true")
    args = parser.parse_args()
    
    if args.latest:
        latest_version = sorted(
            glob('./lightning_logs/version*/'),
            key=lambda x: int(re.match(r".*_([0-9]+)", os.path.dirname(x)).group(1)))[-1]
        ckpts = glob(os.path.join(latest_version, 'checkpoints/') + "*.ckpt")
        print("There are {} ckpt(s) available:".format(len(ckpts)))
        if len(ckpts) > 0:
            for idx, ckpt in enumerate(ckpts):
                print("{}) {}".format(idx, ckpt))
            choice = int(input("Choose: "))
            main(ckpts[choice], args.path2, args.text, args.config_no)
    else:
        main(args.path, args.path2, args.text, args.config_no)
