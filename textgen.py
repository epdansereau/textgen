# Simplified version of https://github.com/huggingface/transformers/blob/master/examples/text-generation/run_generation.py
import torch
import os
import shutil
import random
import pandas as pd
import numpy as np
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from tqdm import trange, tqdm
import argparse
from utils import read_tsv

def set_seed(seed, n_gpu):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)
  

def batch(sentences, labels, batch_size):
    l = len(sentences)
    for ndx in range(0, l, batch_size):
        yield sentences[ndx:min(ndx + batch_size, l)], labels[ndx:min(ndx + batch_size, l)]
        
MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop
def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length

def search_data(term, path):
    '''Used to search the original data'''
    with open(path) as f:
        data = f.read().split("<|endoftext|>")[1:-1]
    return [sent for sent in data if (term in sent)]

class TextGenerator():
    def __init__(self,
                lm_model_path, 
                class_model_path = None, 
                seed=None, 
                device0=0, 
                device1=1, 
                fp16=False, # Not implemented
                num_return_sequences = 8,
            ):
        if seed is not None:
            n_gpu = torch.cuda.device_count()
            set_seed(seed, n_gpu)

        # Initialize the model and tokenizer
        print("Loading language modeling...")
        self.device0 = device0
        self.device1 = device1
        self.num_return_sequences = num_return_sequences
        self.lm_tokenizer = GPT2Tokenizer.from_pretrained(lm_model_path)
        self.lm_model = GPT2LMHeadModel.from_pretrained(lm_model_path)
        self.lm_model.to(device0)
#        if fp16:
#            self.lm_model.half()
        if class_model_path is not None:
            print("Loading classifier...")
            config = AutoConfig.from_pretrained(
                class_model_path,
                num_labels=2,
                finetuning_task=('sst-2',),
                cache_dir=None,
            )
            self.class_tokenizer = AutoTokenizer.from_pretrained(
                class_model_path,
                cache_dir=None,
            )
            self.class_model = AutoModelForSequenceClassification.from_pretrained(
                class_model_path,
                from_tf = False,
                config=config,
                cache_dir=None,
            )
            self.class_model = self.class_model.to(device1)
            self.class_model.eval()
        else:
            self.class_model = None
        print("All models loaded")

    def generate(self, num_return_sequences = 1):
        prompt = ""
        length = 20
        k = 0
        p = 0.9
        temperature = 1.0
        stop_token = "<|endoftext|>"
        repetition_penalty = 1.0  # primarily useful for CTRL model; in that case, use 1.2
        num_beams = 1 # added parameter

        # The max length of the prompt is 1024 for gpt-2
        length = adjust_length_to_model(length, max_sequence_length=self.lm_model.config.max_position_embeddings)

        # Different models need different input formatting and/or extra arguments
        # No preprocessing needed for gpt-2
        encoded_prompt = self.lm_tokenizer.encode("<|endoftext|>" + prompt, add_special_tokens=False, return_tensors="pt")
        encoded_prompt = encoded_prompt.to(self.device0)

        if encoded_prompt.size()[-1] == 0:
            input_ids = None
        else:
            input_ids = encoded_prompt
        output_sequences = self.lm_model.generate(
            input_ids=input_ids,
            max_length=1000,
            temperature=temperature, #temperature of 1.0 has no effect, lower tend toward greedy sampling
            top_k=k,
            top_p=p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            num_return_sequences=num_return_sequences,
            num_beams = num_beams,
            pad_token_id = 50256   #setting this avoids a warning
        ) # A list of tensors with ids

        # Remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        generated_sequences = []

        for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
            generated_sequence = generated_sequence.tolist()[1:] # A list of ids  # removing the first token
            # Decode text
            text = self.lm_tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
            # Remove all text after the stop token
            text = text[: text.find("<|endoftext|>") if stop_token else None]
            # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
            generated_sequences.append(text)

        return generated_sequences

    def _prepare_inputs(self,inputs):
        """
        Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        for k, v in inputs.items():
            inputs[k] = torch.tensor(v).to(self.device1)

    #         if self.args.past_index >= 0 and self._past is not None:
    #             inputs["mems"] = self._past

        return inputs

    def batch_to_ids(self, batch):
        max_length = 128
        return self.class_tokenizer(batch, padding = 'max_length', truncation = True, max_length = max_length)

    def eval_batch(self, text_batch):
        ids = self.batch_to_ids(text_batch)
        self._prepare_inputs(ids)
        return self.class_model(**ids)[0].detach()

    def choose(self, batch, treshold):
        outputs = self.eval_batch(batch)
        good = outputs[:,1] > treshold
    #     if good.any():
    #         return batch[torch.where(good)[0]]
        for index in torch.where(good)[0]:
            return batch[index]

    def _generate_selected(self, treshold = 2.5):
        for _ in range(100000):
            try:
                good = self.choose(self.generate(self.num_return_sequences), treshold)
            except RuntimeError:
                print("Runtime error") # Tolerating OOM errors
                good = False
            if good:
                return good

    def generate_selected(self, amount, path = None, treshold  = 2.5):
        if path is None:
            for _ in trange(amount):
                print(self._generate_selected(treshold))
        else:
            if os.path.exists(path):
                print("Output will be added to already existing file {}".format(path))
            with open(path, "a") as f:
                for _ in trange(amount):
                    f.write(self._generate_selected(treshold))
                    f.write("<|endoftext|>")

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Generate texts from models.')
    parser.add_argument('--lm_model_path', type=str, help='Path of the language modeling model')
    parser.add_argument('--class_model_path', type=str, help='Path of the language modeling model')
    parser.add_argument('--amount', type=int, help='Amount of texts to generate', default=1)
    parser.add_argument('--treshold', type=float, help='Mininum score of selected texts', default=2.2)
    parser.add_argument('--output_path', type=str, help='Path where outputs will be saved (if not specified, output will be printed)', default=None)
    parser.add_argument('--seed', type=int, help='Set seed (optional)', default=None)
    parser.add_argument('--num_return_sequences', type=int, help='Size of batch of generated texts', default=8)

    args = parser.parse_args()

    textgen = TextGenerator(args.lm_model_path, args.class_model_path, seed = args.seed, num_return_sequences = args.num_return_sequences)

    textgen.generate_selected(args.amount, path = args.output_path, treshold = args.treshold)

