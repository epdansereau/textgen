python subreddit_text_downloader.py --subreddit writingprompts \
  --start_time 1607824350 \
  --end_time 1607997150
python reddit_data_cleaning.py --data_source fetched_r_writingprompts_1607824350_1607997150.json \
  --output_dir lm_data \
  --required_in_title [wp]
python gpt2xl_lm.py --model_source gpt2-xl \
  --batch_size=6 \
  --output_dir lm_model \
  --train_path lm_data/train.txt \
  --eval_path lm_data/train.txt \
  --epochs=1 \
  --log_steps=200 \
  --learning_rate=5e-6 \
  --fp16 \
  --gradient_checkpointing \
  --split_offset=-1
mv lm_model/*/ lm_model/test_model
python adversarial_dataset.py --train_data lm_data/train.txt \
  --test_data lm_data/test.txt \
  --val_data lm_data/dev.txt \
  --output_dir class_data \
  --model lm_model/test_model
python run_glue.py \
  --model_name_or_path roberta-large-openai-detector \
  --task_name SST-2 \
  --data_dir class_data \
  --do_train \
  --learning_rate 5e-6 \
  --num_train_epochs 1.0 \
  --output_dir=class_model
python textgen.py --lm_model_path lm_model/test_model \
  --class_model_path class_model \
  --amount=1 \
  --treshold=-5  