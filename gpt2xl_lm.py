from transformers import DataCollatorForLanguageModeling, AutoTokenizer, AutoConfig

import torch
from transformers import TextDataset
from torch.utils.data.dataloader import DataLoader
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from tqdm import tqdm, trange

from torch.cuda.amp import autocast

import os
from os.path import join
from os import getcwd
from time import time
import warnings
import math
import argparse
from utils import delete_dir

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import GPT2PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPastAndCrossAttentions, BaseModelOutputWithPastAndCrossAttentions
from transformers.modeling_gpt2 import Block


class GPT2Model(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # custom settings:
        self.gradient_checkpointing = config.gradient_checkpointing
        self.split_offset = config.split_offset

        self.wte = nn.Embedding(config.vocab_size, config.n_embd).to(0)
        
        self.wpe = nn.Embedding(config.n_positions, config.n_embd).to(0)
        self.drop = nn.Dropout(config.embd_pdrop).to(0)
        self.h = nn.ModuleList([Block(config.n_ctx, config, scale=True) for _ in range(config.n_layer)])
        self.split_point = int(config.n_layer/2) + self.split_offset
        for module in range(self.split_point):
            self.h[module].to(0)
        for module in range(self.split_point, config.n_layer):
            self.h[module].to(1)
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon).to(1)
        

        self.init_weights()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]

        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = [None] * len(self.h)
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tenBaseModelOutputWithPastAndCrossAttentionsds, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

#             if getattr(self.config, "gradient_checkpointing", False):
            if self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # checkpointing only works with tuple returns, not with lists
                        return tuple(output for output in module(*inputs, use_cache, output_attentions))

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    layer_past,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
            
            

            if i == self.split_point - 1:
                if self.gradient_checkpointing:
                    outputs = list(outputs)
                    for j in range(len(outputs)):
                        outputs[j] = outputs[j].to(1)
                    outputs = tuple(outputs)
                else:
                    for j in range(len(outputs)):
                        outputs[j] = outputs[j].to(1)
                if use_cache:
                    presents = list(presents)
                    for j in range(len(presents)):
                        presents[j] = presents[j].to(1)
                    presents = tuple(presents)
                if output_attentions:
                    all_self_attentions = list(all_self_attentions)
                    all_cross_attentions = list(all_cross_attentions)
                    for j in range(len(all_self_attentions)):
                        all_self_attentions[j] = all_self_attentions[j].to(1)
                    for j in range(len(all_cross_attentions)):
                        all_cross_attentions[j] = all_cross_attentions[j].to(1)

            hidden_states, present = outputs[:2]
            if use_cache is True:
                presents = presents + (present,)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2],)
                all_self_attentions = all_cross_attentions + (outputs[3],)

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )




class GPT2LMHeadModel(GPT2PreTrainedModel):
    authorized_missing_keys = [r"h\.\d+\.attn\.masked_bias", r"lm_head\.weight"]

    def __init__(self, config):
        super().__init__(config)

        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.init_weights()
        
        self.lm_head = self.lm_head.to(0)

    def get_output_embeddings(self):
        return self.lm_head

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
        

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        """
        
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        transformer_outputs = list(transformer_outputs)
        for i in range(len(transformer_outputs)):
            if isinstance(transformer_outputs[i], tuple):
                transformer_outputs[i] = list(transformer_outputs[i])
                for j in range(len(transformer_outputs[i])):
                    transformer_outputs[i][j] = transformer_outputs[i][j].to(0)
                transformer_outputs[i] = tuple(transformer_outputs[i])
            else:
                transformer_outputs[i] = transformer_outputs[i].to(0)
        transformer_outputs = tuple(transformer_outputs)
        
        hidden_states = transformer_outputs[0]
        
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        
        
        return CausalLMOutputWithPastAndCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

scaler = torch.cuda.amp.GradScaler()

def load_model(model_source,
            learning_rate = None,
            gradient_checkpointing = False,
            split_offset=0):
    config = AutoConfig.from_pretrained(model_source)
    if learning_rate is not None:
        config.learning_rate = learning_rate

    # custom settings:
    config.gradient_checkpointing = gradient_checkpointing
    config.split_offset = split_offset

    model = GPT2LMHeadModel.from_pretrained(model_source, config = config)
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    return model, tokenizer

def get_dataloaders(model, tokenizer, batch_size, train_path, eval_path):
    block_size = 1024
    train_dataset = TextDataset(tokenizer=tokenizer,
                      file_path=train_path,
                      block_size=block_size)
    test_dataset = TextDataset(tokenizer=tokenizer,
                          file_path=eval_path,
                          block_size=block_size)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False,mlm_probability=0.15)
    trainloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        drop_last=False,
        num_workers=0,
    )
    testloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        drop_last=False,
        num_workers=0,
    )
    return trainloader, testloader

def _prepare_inputs(inputs, device = 0):
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)

def create_optimizer_and_scheduler(model, dataloader, epochs, learning_rate):
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_epsilon = 1e-08
    weight_decay = 0.0
    warmup_steps=0
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        eps=adam_epsilon,
    )
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs*len(dataloader)
    )
    
    return optimizer, lr_scheduler

def evaluate(dataloader, fp16=True, device = 0):
    model.eval()
    
    total_loss = 0
    print("evaluating")
    for step, inputs in enumerate(tqdm(dataloader), 1):
        _prepare_inputs(inputs, device)
        
        with torch.no_grad():
            if fp16:
                with autocast():
                    total_loss += model(**inputs)[0]
            else:
                total_loss += model(**inputs)[0]
    return total_loss/len(dataloader)

def train_step(model, inputs):
    loss = model(**inputs)[0]
    loss.backward()
    return loss.detach()

def optimizer_step(model, optimizer):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm = 1.0)
    optimizer.step()

def fp16_train_step(model, inputs):
    with autocast():
        loss = model(**inputs)[0]
    scaler.scale(loss).backward()
    return loss.detach()
    
def fp16_optimizer_step(model, optimizer, max_grad_norm = 1.0):
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    scaler.step(optimizer)
    scaler.update()
    
def save(model, dir_name, optimizer, scheduler, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    path = join(output_dir, dir_name)
    os.makedirs(path)

    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    torch.save(optimizer.state_dict(), join(path, "optimizer.pt"))
    torch.save(scheduler.state_dict(), join(path, "scheduler.pt"))
    print("model saved at", join(getcwd(), path))


def train(dataloader,
        output_dir,
        epochs=6,
        log_steps=200,
        learning_rate=5e-6,
        fp16=True,
        debug_stop = False,
        device = 0, 
        optimizer = None,
        lr_scheduler = None,
        ):
    if optimizer is None:
        optimizer, lr_scheduler = create_optimizer_and_scheduler(model, 
                                    dataloader,
                                    epochs,
                                    learning_rate=learning_rate)
    delete_dir(output_dir)
    for epoch in range(epochs):
        total_loss = 0
        for step, inputs in enumerate(tqdm(dataloader), 1):
            model.train()
            _prepare_inputs(inputs, device)

            if fp16:
                total_loss += fp16_train_step(model, inputs)/log_steps
                fp16_optimizer_step(model, optimizer)
            else:
                total_loss += train_step(model, inputs)/log_steps
                optimizer_step(model, optimizer)

            lr_scheduler.step()

            model.zero_grad()

            # Logging
            if not((step)%log_steps):
                print(f"step: {step} (lr = {optimizer.param_groups[0]['lr']}), loss: {total_loss}")
                
                total_loss = 0
                if debug_stop:
                    break
        eval_loss = evaluate(testloader, fp16=fp16, device=0)
        perplexity = float(math.exp(eval_loss))
        print("perplexity:", perplexity)
        save_dir = '{}_ep{}_perplexity{}'.format(int(time()), epoch, perplexity)
        save(model, save_dir, optimizer, lr_scheduler, output_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_source', type=str)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--train_path', type=str)
    parser.add_argument('--eval_path', type=str)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--log_steps', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=5e-6)
    parser.add_argument('--split_offset', type=int, default=-1)
    parser.add_argument('--fp16', default=False, action='store_true')
    parser.add_argument('--gradient_checkpointing', default=False, action='store_true')
    parser.add_argument('--debug_stop', default=False, action='store_true')
    parser.add_argument('--device', type=int, default=0)


    args = parser.parse_args()

    model, tokenizer = load_model(args.model_source,
                        learning_rate = args.learning_rate,
                        gradient_checkpointing=args.gradient_checkpointing,
                        split_offset = args.split_offset,
                        )
    print("Model loaded")

    trainloader, testloader = get_dataloaders(model,
            tokenizer,
            batch_size = args.batch_size,
            train_path = args.train_path,
            eval_path = args.eval_path)
    print("Data loaded")

    train(trainloader,
            output_dir = args.output_dir,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            log_steps=args.log_steps,
            fp16=args.fp16,
            debug_stop = args.debug_stop,
            device=args.device
            )