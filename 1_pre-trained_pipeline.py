import warnings
from transformers import pipeline

# 警告を無視する設定
warnings.filterwarnings("ignore", category=FutureWarning)

# DistilBERTモデルを使って質問応答を行う
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

context = "Large language models can process natural language tasks."
question = "What can large language models process?"

result = qa_pipeline(question=question, context=context, clean_up_tokenization_spaces=False)
print("こまん：", result['answer'])
