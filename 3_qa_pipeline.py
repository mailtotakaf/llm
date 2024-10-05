from transformers import pipeline

# "question-answering"タスクに使う事前学習済みモデルをロード
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# コンテキスト（背景情報）と質問を設定
context = "Large language models like BERT can process various natural language tasks effectively."
question = "What can BERT process?"

# 質問に対して応答を得る
result = qa_pipeline(question=question, context=context)

# 結果を表示
print(f"Answer: {result['answer']}")
