from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import torch

# 1. IMDbデータセットのロード
dataset = load_dataset("imdb")

# 2. モデルとトークナイザーの準備
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# 3. テキストをモデルに入力できる形式にトークナイズ
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 訓練・テストデータの準備（データ量を減らしている）
# train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(10000))
# test_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(2000))

# 学習データセット全体を使用すると時間がかかるため、まずはサブセット（小さいデータセット）を使って動作確認を行います。
train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))  # データ数を1000件に削減
test_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(200))  # テストデータも縮小

# 4. トレーニング設定
# training_args = TrainingArguments(
#     output_dir="./results",          # 出力ディレクトリ
#     evaluation_strategy="epoch",     # エポックごとに評価
#     per_device_train_batch_size=8,   # 訓練時のバッチサイズ
#     per_device_eval_batch_size=8,    # 評価時のバッチサイズ
#     num_train_epochs=3,              # エポック数
#     weight_decay=0.01,               # 重みの減衰
# )

# バッチサイズが大きいとメモリ消費が増え、処理速度が低下することがあります。小さめのバッチサイズで試してみる。
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,   # バッチサイズを4に減らす
    per_device_eval_batch_size=4,
    num_train_epochs=1,  # とりあえず1～3で動作確認
    evaluation_strategy="epoch",
)

# トレーナーの設定
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# 5. モデルの訓練と評価
trainer.train()
trainer.evaluate()

# 量子化
quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

# 量子化後のモデルの保存
torch.save(quantized_model.state_dict(), "quantized_distilbert.pth")

# モデルの保存(GPU環境でトレーニングする用)
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")
