import concurrent.futures
from transformers import pipeline

# 事前学習済みモデルのロード
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# コンテキスト（背景情報）と複数の質問を設定
context = "Large language models like BERT can process various natural language tasks effectively."
questions = [
    "What can BERT process?",
    "What is BERT?",
    "What tasks can BERT handle?"
]

# 質問応答の処理を並列化
def process_question(question):
    return qa_pipeline(question=question, context=context)

# 並列処理で複数の質問に回答
with concurrent.futures.ThreadPoolExecutor() as executor:
    results = list(executor.map(process_question, questions))

# 結果を表示
for i, result in enumerate(results):
    print(f"Question {i+1}: {questions[i]}")
    print(f"Answer: {result['answer']}")
    print()