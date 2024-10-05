from transformers import pipeline

# テキスト分類の事前学習済みモデルをロード
classifier = pipeline("sentiment-analysis")

# テキスト分類を行う
result = classifier("I like sushi.")

# 結果を表示
print(result)
