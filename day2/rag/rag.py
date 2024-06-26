from langchain_community.chat_models import ChatOllama
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain import hub
import os


embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


def initial_vectorstore():
  # load web content
  loader = WebBaseLoader([
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
  ]);
  docs = loader.load()


  # split docs
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200,
    length_function=len
  )
  splited_docs = text_splitter.split_documents(docs)

  # save to disk
  db = Chroma.from_documents(splited_docs, embeddings, persist_directory="./chroma_db")
  return db




# load from disk or create db
if os.path.isdir("./chroma_db"):
  db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
else:
  db = initial_vectorstore()


# setting retriever
retriever = db.as_retriever()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def relevance_checker(query):

  # Define your desired data structure.
  class Relevance(BaseModel):
      relevance: str = Field(description="result of relevance check. 'YES' or 'NO'")


  parser = JsonOutputParser(pydantic_object=Relevance)

  checker_llm = ChatOllama(model='llama3', temperature=0)
  check_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are given a user query and a retrieved document. 
    Determine if the retrieved document is relevant to the user's query. 
    Respond with "YES" if the document is relevant and "NO" if it is not. 
    Consider the context and specificity of the user's query when making your decision.

    User Query: {question}

    Retrieved Document: {context}

    Is the retrieved document relevant to the user's query?
    {format_instructions}
    """,
    partial_variables={"format_instructions": parser.get_format_instructions()}
  )
  checker_chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()}
  | check_prompt
  | checker_llm
  | parser)
  return checker_chain.invoke(query)

def hallucination_checker(query, answer):

  # Define your desired data structure.
  class Hallucination(BaseModel):
      hallucination: str = Field(description="result of hallucination check. 'YES' or 'NO'")

  checker_llm = ChatOllama(model='llama3', temperature=0)
  parser = JsonOutputParser(pydantic_object=Hallucination)
  check_prompt = PromptTemplate(
    input_variables=["context", "question", "answer"],
    template="""You are given a user query and a response generated by the RAG system. 
    Determine if the response contains hallucinated information, meaning it includes details or facts not supported by the retrieved documents. 
    Respond with "YES" if the response is hallucinated and "NO" if it is accurate and supported by the retrieved documents.

    User Query: {question}

    Retrieved Documents: {context}

    RAG System Response: {answer}

    Is the response hallucinated?
    {format_instructions}
    """,
    partial_variables={"format_instructions": parser.get_format_instructions()}
  )
  check_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough(), "answer": lambda x: answer}
    | check_prompt
    | checker_llm
    | parser
  )
  return check_chain.invoke(query)



def run_rag(query):
  llm = ChatOllama(model='llama3', temperature=0)
  prompt = hub.pull("rlm/rag-prompt")
  rag_chain = (
      {"context": retriever | format_docs, "question": RunnablePassthrough()}
      | prompt 
      | llm
      | StrOutputParser()
  )

  return rag_chain.invoke(query)


retry_count = 0
def run_rag_with_hallucination_check(query):
  answer = run_rag(query)
  hallucination_result = hallucination_checker(query, answer)
  if (hallucination_result['hallucination'] == 'NO'):
    print(answer)
  elif retry_count > 2:
    print('exceed retry count')
    return
  else:
    retry_count = retry_count+1
    run_rag_with_hallucination_check(query)



query = "What is Chain-of-Thought?"
check_result = relevance_checker(query)

if check_result['relevance'] == 'YES':
  run_rag_with_hallucination_check(query)
else:
  print(check_result) # print NO