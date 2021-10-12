# AI Quarreler

Using PTT Gossiping articles as dataset, to train an AI Quarreler who is able to respond 問卦標題 properly. **Made For Fun!**

## Prerequisite

- Python 3.7.*
- Python Package
  - Training
    - gitpython
    - tensorboard>=2.2
    - torch==1.7.1
    - transformers>=4.2.2
    - pytorch-nlp
    - pytorch-lightning>=1.2.2
  - API
    - fastapi
    - uvicorn[standard]

For the latest package requirements, please refer to [requirements.txt](requirements.txt).

## Installation

    git clone https://github.com/qqaatw/AIQuarreler.git

## Building dataset

Steps:

1. Use [crawler](https://github.com/jwlin/ptt-web-crawler) to crawl articles in json format.
2. Use dataset builder to build tsv dataset.
    ```
    python dataset_builder.py -p [json file path] \
      --included_categories 問卦 \
      --min_fetched_pushes 10
    ```

## Training

    # Default training config. 
    # --dataset_name can be specified multiple times to add more datasets.
    python torch_train.py --max_epochs 50 \
      --dataset_name [filename of tsv dataset] \
      --batch_size 16 \
      --model_name bert-base-chinese \
      --num_workers 20 \
      --learning_rate 1e-5 \
      --dataset_split_ratio 0.9 \
      --git_record \
      --gpus 1, \
      --gradient_clip_val 1

    # Use the following command to view available options: 
    python torch_train.py --help

## Generation

A pretrained checkpoint can be found [here](https://drive.google.com/file/d/1RsnA-TphQpvjUVzd7qByY70HSW2Pxnn5/view?usp=sharing).

    # An example of text generation.
    python torch_generate.py -t 有沒有高雄市政府的八卦？ --path last_version3.ckpt

    # Use the following command to view available options: 
    python torch_generate.py --help

Here are a number of pre-defined generation strategies you can try it out, by specifying `--config_no`:
    
0. Greedy search, meaning that the generator will always choose the most possible character on each auto-regressive step.
1. Beam search with num_beams = 2, meaning that the generator will keep tracking up to 2 beams and return the highest probable sequence.
2. Beam search with num_beams = 3, meaning that the generator will keep tracking up to 3 beams and return the highest probable sequence.
3. Beam search with num_beams = 4, meaning that the generator will keep tracking up to 4 beams and return the highest probable sequence.
4. Sampling from most likely 5 words. The result may vary on each run.
5. Sampling from the minimum number of words to exceed 92 % of the probability mass. The result may vary on each run.

```
# Choouse No.2 strategy.
python torch_generate.py -t 有沒有高雄市政府的八卦？ --path last_version3.ckpt --config_no 2
```

## API
    
    # Bind 0.0.0.0 to make API accessible to local network.

    uvicorn api:app --reload --host 0.0.0.0

    # API docs

    http://ipaddress:8000/docs