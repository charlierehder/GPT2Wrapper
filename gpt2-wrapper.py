import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('microsoft/DialoGPT-medium')
model = GPT2LMHeadModel.from_pretrained('microsoft/DialoGPT-medium')

history_tokens = None
while True:

    input_text = input('Me: ') + tokenizer.eos_token
    if input_text == "quit" + tokenizer.eos_token:
        break

    user_tokens = tokenizer.encode(input_text, return_tensors='pt')

    model_input_tokens = user_tokens if not history_tokens else torch.cat([history_tokens, user_tokens], dim=-1)
    model_output = model.generate(model_input_tokens, max_length=200, pad_token_id=tokenizer.eos_token_id) 

    output_text = tokenizer.decode(model_output[:, model_input_tokens.shape[-1]:][0], skip_special_tokens=True)
    print("GPT: " + output_text)
