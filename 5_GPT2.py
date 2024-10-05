from transformers import GPTNeoForCausalLM, GPT2Tokenizer

# Use GPTNeoForCausalLM instead of GPT2LMHeadModel
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

prompt = "please show me some money."
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(inputs["input_ids"], max_length=50)
print("出力：", tokenizer.decode(outputs[0], skip_special_tokens=True))
