import os
from dotenv import load_dotenv
from llm_utils import llm_judge_answer, llm_predict_answer
from utils import load_longmemeval_o_df

load_dotenv()

N = 10
correct_predictions = 0
longmemeval_o_df = load_longmemeval_o_df().head(N)

for index, row in longmemeval_o_df.iterrows():
    question = row["question"]
    answer = row["answer"]

    sessions = row["haystack_sessions"]
    evidence = ""
    for session in sessions:
        for message in session:
            evidence += f"{message['role']}: {message['content']}\n"

    predicted_answer = llm_predict_answer(question, evidence)
    answer_is_correct = llm_judge_answer(question, predicted_answer, answer)
    if answer_is_correct:
        correct_predictions += 1

    print(f"Question: {question}")
    print(f"Predicted Answer: {predicted_answer}")
    print(f"Answer: {answer}")
    print(f"Correct: {answer_is_correct}")
    print("-" * 100)

print(f"Correct predictions: {correct_predictions}/{N}")
print(f"Accuracy: {correct_predictions/N}")