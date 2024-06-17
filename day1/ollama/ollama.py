import requests

mathRole = """
your role is math teacher.
you can solve math problem.
you answer step by step how to solve the problem and explain the answer.
"""

historyRole = """
your role is history teacher.
the role is very well know about history.
you answer very kindly.
"""

system = f"""
there is three role you know.

#role description.

## Math
{mathRole}

## History
{historyRole}


user ask question. then you must answer what role can answer the user's question.
only answer "Math" or "History" as above role description.
before answer, think about question and which role is right to answer step by step.

here is some example.

user) what is the Newton’s law?
thought) Newton's laws are laws used in physics. Physics is related to mathematics because it is a problem that can be solved with mathematics.
anwer) Math

user) How did World War I end?
thought) World War I is an event that happened in the past. It is related to history because it was a war that had a great impact on history.
answer) History
"""

question = "when did Korean war happen?"

res = requests.post('http://localhost:11434/api/generate', json={
  "model": "llama3",
  "system": system,
  "prompt": question,
  "stream": False
}).json()
role = res.get('response')

print(role)
print('--------------')

answerSystem = ''
if role == "History":
  answerSystem = historyRole
elif role == "Math":
  answerSystem = mathRole

answer = requests.post('http://localhost:11434/api/generate', json={
  "model": "llama3",
  "system": answerSystem,
  "prompt": question,
  "stream": False
}).json()
print (answer.get('response'))