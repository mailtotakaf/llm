from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# トレーニング済みモデルとトークナイザーのロード
model = AutoModelForSequenceClassification.from_pretrained("./trained_model")
tokenizer = AutoTokenizer.from_pretrained("./trained_model")

# 量子化（オプション：推論を高速化）
quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

# テキストのトークナイズ
inputs = tokenizer("your dick is so smelly", return_tensors="pt")

# CPU上で推論実行
with torch.no_grad():
    outputs = quantized_model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

print(f"Prediction: {predictions}")
