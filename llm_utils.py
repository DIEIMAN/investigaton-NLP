from litellm import completion


def llm_judge_answer(question, predicted_answer, answer, model="azure/gpt-4.1"):
    prompt = f"""
    You are a helpful assistant that judges the correctness of an answer to a question.
    The question is: {question}
    The prediction is: {predicted_answer}
    The answer is: {answer}
    Return True if the prediction is correct, False otherwise. No other text or explanation.
    """
    response = completion(model=model, messages=[{"role": "user", "content": prompt}])
    return eval(response.choices[0].message.content)


def llm_predict_answer(question, evidence, model="azure/gpt-4.1"):
    prompt = f"""
    You are a helpful assistant that answers a question based on the evidence.
    The evidence is: {evidence}
    The question is: {question}
    Return the answer to the question.
    """
    response = completion(model=model, messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content
