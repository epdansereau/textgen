# Prompts

This repo contains the code to generate story prompts as showcased on https://myths.ai/prompts .

## Setup
This project is built on the transformers library from HuggingFace, v.3.4.0 (https://github.com/huggingface/transformers)

The gpt2-xl model is fine-tuned for language modeling, and roberta-large is fine-tuned as a classifer.

Two 24GB gpus with cuda are needed to run the code as is.


## How to use

The data is fetched from reddit.com/r/writing_prompts . To scrape the whole subreddit, we use the free pushshift.io API. Consider donating to them if you use this service a lot.

Reddit API keys need to be provided for the download script. These are easy to obtain, instructions here : https://praw.readthedocs.io/en/latest/getting_started/authentication.html)
```
python subreddit_text_downloader.py --client_id YourClientId \
  --client_secret YourClientSecret \
  --user_agent YourUserAgent \
  --subreddit writingprompts
```

The data is downloaded and saved as a list of json, saved with a name like *fetched_r_writingprompts_1283899940_1606950225.json* . The data is then cleaned and separated into train/test/dev sets. Here we only keep posts with the string "[wp]" (case insensitive) in the title, and that have been upvoted at least once. We exclude posts with the strings "write " and "describe " in them, and we skip the first 1000 posts.
```
python reddit_data_cleaning.py --data_source fetched_r_writingprompts_1283899940_1606950225.json \
    --output_dir lm_data \
    --upvoted \
    --required_in_title [wp] \
    --bad_words write describe \
    --skip_first 1000
```

Gpt2-xl for language modeling is finetuned on the dataset. The model is splitted between 2 24Gb gpus:
```
python gpt2xl_lm.py --model_source gpt2-xl \
  --batch_size=6 \
  --output_dir lm_model \
  --train_path lm_data/train.txt \
  --eval_path lm_data/dev.txt \
  --epochs=6 \
  --log_steps=200 \
  --learning_rate=5e-6 \
  --fp16 \
  --gradient_checkpointing \
  --split_offset=-1
```

We then fine-tune a classifier model that learns to differentiate generated prompts from real ones. The prompts that can fool the classifier will then be selected as valid prompts.  
```
python adversarial_dataset.py --train_data lm_data/train.txt \
  --test_data lm_data/test.txt \
  --val_data lm_data/dev.txt \
  --output_dir class_data \
  --model lm_model/1606969116_ep4_perplexity19.491223080778077
```

A classifier is trained using the example run_glue.py script provided by hugging face:
```
python run_glue.py \
  --model_name_or_path roberta-large-openai-detector \
  --task_name SST-2 \
  --data_dir class_data \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --learning_rate 5e-6 \
  --num_train_epochs 3.0 \
  --output_dir=class_model \
  --logging_steps=400 \
  --save_steps=30250 \
  --eval_steps=30250
```

Prompts can then be generated using both models, a generator and a classifier that chooses hopefully the best ones. If an --output_dir is provided, the results will be saved in a file, otherwise they will be printed. 
```
python textgen.py --lm_model_path lm_model/1606969116_ep0_perplexity44.77411841639417 \
  --class_model_path class_model \
  --amount=8 \
  --treshold=2.7  
```
