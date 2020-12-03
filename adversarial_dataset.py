import torch
import os
from os.path import join
import shutil
import random
import argparse
        
from tqdm import trange, tqdm

import pandas as pd

# Simplified version of https://github.com/huggingface/transformers/blob/master/examples/text-generation/run_generation.py

import logging

import numpy as np
import torch

from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoConfig

from utils import delete_dir

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop



def set_seed(seed, n_gpu):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length
        
        
def load_model(model_name_or_path = "gpt2",
               seed=42, 
               fp16 = False, #Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit
               device = 0
              ):
    n_gpu = torch.cuda.device_count()
    logger.warning(
        "device: %s, n_gpu: %s, 16-bits training: %s",
        device,
        n_gpu,
        fp16,
    )
    set_seed(seed, n_gpu)

    # Initialize the model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
    config = AutoConfig.from_pretrained(model_name_or_path)
    config.pad_token_id = config.eos_token_id
    model = GPT2LMHeadModel.from_pretrained(model_name_or_path, config = config)
    print(n_gpu)
    model.to(device)

    if fp16:
        model.half()
        
    return model, tokenizer

def generate_text(model,
                  tokenizer,
                  prompt = "",
                  length = 20,
                  num_return_sequences = 1,
                  k = 0,
                  p = 0.9,
                  temperature = 1.0,
                  stop_token = "<|endoftext|>",
                  repetition_penalty = 1.0,  # primarily useful for CTRL model; in that case, use 1.2
                  num_beams = 1, # added parameter
                  device = 0,
                 ):

    # The max length of the prompt is 1024 for gpt-2
    length = adjust_length_to_model(length, max_sequence_length=model.config.max_position_embeddings)

    # Different models need different input formatting and/or extra arguments
    # No preprocessing needed for gpt-2
    encoded_prompt = tokenizer.encode("<|endoftext|>" + prompt, add_special_tokens=False, return_tensors="pt")
    encoded_prompt = encoded_prompt.to(device)

    if encoded_prompt.size()[-1] == 0:
        input_ids = None
    else:
        input_ids = encoded_prompt
    output_sequences = model.generate(
        input_ids=input_ids,
        min_length = 2,
        max_length=length + len(encoded_prompt[0]),
        temperature=temperature, #temperature of 1.0 has no effect, lower tend toward greedy sampling
        top_k=k,
        top_p=p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        num_return_sequences=num_return_sequences,
        num_beams = num_beams,
    ) # A list of tensors with ids

    # Remove the batch dimension when returning multiple sequences
    if len(output_sequences.shape) > 2:
        output_sequences.squeeze_()

    generated_sequences = []

    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        generated_sequence = generated_sequence.tolist()[1:] # A list of ids  # removing the first token
        # Decode text
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
        # Remove all text after the stop token
        text = text[: text.find("<|endoftext|>") if stop_token else None]
        # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
        generated_sequences.append(text)

    return generated_sequences

def create_data(output_dir, source_train, source_test, source_dev, gen_batch = 64, length = 128, device = 0):
    delete_dir(output_dir)
    os.makedirs(output_dir)
    for datatype, source_path in zip(["train", "test", "dev"],[source_train, source_test, source_dev]):
        with open(source_path) as f:
            data = f.read()    
        data = data.split("<|endoftext|>")
        data = data[1:-1]
        len_ = len(data)
        data = [text.replace("\n","").replace("\t","") for text in data]
        steps = int(len_/gen_batch) + 1

        texts = []

        for _ in trange(steps):
            texts += generate_text(model,
                                  tokenizer,
                                  prompt = "",
                                  length = length,
                                  num_return_sequences = gen_batch,
                                  device = device,
                                 )
        texts = texts[:len_]
        
        # Cleaning the strings:
        texts = [text.replace("\n","").replace("\t","") for text in texts]

        total_data = []
        labels = [0]*len_ + [1]*len_

        import random
        random.shuffle(labels)
        for label in labels:
            if label:
                total_data.append(data.pop())
            else:
                total_data.append(texts.pop())
        df = pd.DataFrame({"sentence":total_data, "label":labels})
        if datatype == "test":
            df["sentence"].to_csv(join(output_dir,"test.tsv"), sep = "\t", index_label = "index")
            df["label"].to_csv(join(output_dir,"test_answers.tsv"), sep = "\t", index_label = "index")
        else:
            df.to_csv(join(output_dir,datatype +".tsv"), sep = "\t", index = False)
                          

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--test_data', type=str)
    parser.add_argument('--val_data', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--gen_batch', type=int, default=64)
    parser.add_argument('--length', type=int, default=128)

    args = parser.parse_args()

    model, tokenizer = load_model(args.model, device = args.device)

    create_data(args.output_dir,
                args.train_data,
                args.test_data,
                args.val_data,
                device = args.device,
                gen_batch = args.gen_batch,
                length = args.length)
