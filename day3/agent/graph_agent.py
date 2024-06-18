from langchain_community.chat_models import ChatOllama
from pprint import pprint
from typing import List
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain import hub
from tavily import TavilyClient
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph
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


def documents_to_str(documents):
  context = [{"content": obj.page_content, "source": obj.metadata["source"]} for obj in documents]
  return "\n\n".join(list(map(lambda doc: f'{doc["content"]}\nsource: {doc["source"]}', context)))


# docs_retrieval 
def docs_retrieval(state):
  question = state["question"]
  documents = retriever.invoke(question)
  return {"documents": documents, "question": question, "retry_hallucination_count": 0}


def relevance_check(state):
  question = state["question"]
  documents = state["documents"]


  checker_llm = ChatOllama(model='llama3', format="json", temperature=0)
  check_prompt = PromptTemplate(
    input_variables=["documents", "question"],
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are given a user query and a retrieved document. 
    Determine if the retrieved document is relevant to the user's query. 
    Respond with "YES" if the document is relevant and "NO" if it is not. 
    Consider the document and specificity of the user's query when making your decision.

    Return the a JSON with a single key 'relevance' and no premable or explanation.

    ### User Query: 
    {question}

    ### Retrieved Document: 
    {documents}
    <|eot_id|>
    <|start_header_id|>user<|end_header_id|>Is the retrieved document relevant to the user's query?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
  )
  checker_chain = ( 
    check_prompt
  | checker_llm
  | JsonOutputParser())

  check_result = checker_chain.invoke({"question": question, "documents": documents_to_str(documents)})

  relevance = check_result["relevance"]
  return {"question": question, "documents": documents, "relevance": relevance}


def search_travily(state):
  question = state["question"]
  documents = state["documents"]

  tavily = TavilyClient(api_key='')
  docs = tavily.search(query=question, max_results=3)["results"]
  web_results = [Document(d["content"], metadata={"source": d["url"]}) for d in docs]

  documents = web_results

  return {"question": question, "documents": documents}

def generate_answer(state):
  question = state["question"]
  documents = state["documents"]

  llm = ChatOllama(model='llama3')
  prompt = hub.pull("rlm/rag-prompt")
  prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="""
      <|begin_of_text|><|start_header_id|>system<|end_header_id|>You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
      Please mention at last of answer the source information of the documents you referred to.

      Question: {question} 

      Context: {context} 

      Answer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """

  )
  rag_chain = (
      prompt 
      | llm
      | StrOutputParser()
  )

  answer = rag_chain.invoke({"question": question, "context": documents})
  return {"question": question, "documents": documents, "answer": answer}

def hallucination_check(state):
  question = state["question"]
  answer = state["answer"]
  documents = state["documents"]
  retry_hallucination_count = state["retry_hallucination_count"]
  print(f'retry count: {retry_hallucination_count}')

  checker_llm = ChatOllama(model='llama3', format="json", temperature=0)
  check_prompt = PromptTemplate(
    input_variables=["documents", "answer"],
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are given a user query and a response generated by the RAG system. 
    Determine if the response contains hallucinated information, meaning it includes details or facts not supported by the retrieved documents. 
    Respond with "YES" if the response is hallucinated and "NO" if it is accurate and supported by the retrieved documents.
    don't be too strict.

    Return the a JSON with a single key 'hallucination' and no premable or explanation.

    ### Retrieved Documents: 
    {documents}

    ### RAG System Response: 
    {answer}<|eot_id|>
    <|start_header_id|>user<|end_header_id|>Is the response hallucinated?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
  )
  check_chain = (
    check_prompt
    | checker_llm
    |JsonOutputParser() 
  )
  check_result = check_chain.invoke({"documents": documents_to_str(documents), "answer": answer})
  has_hallucination = check_result["hallucination"]
  return {"documents": documents, "answer": answer, "question": question, "hallucination": has_hallucination, "retry_hallucination_count": retry_hallucination_count+1}

class GraphState(TypedDict):
  """
  Represents the state of our graph.

  Attributes:
      question: question
      generation: LLM generation
      web_search: whether to add search
      documents: list of documents
  """

  question: str
  answer: str
  documents: List[Document]
  hallucination: str
  retry_hallucination_count: int
  relevance: str


def relevance_check_v_generate_answer_and_search_travily(state):
  relevance = state["relevance"]

  if relevance == 'YES':
    return "generate"
  else:
    return "travily"

def hallucination_check_v_generate_answer_and_end(state):
  hallucination = state["hallucination"]
  retry_hallucination_count = state["retry_hallucination_count"]
  if hallucination == "YES" and retry_hallucination_count < 10:
    return "generate"
  else:
    return "end"



workflow = StateGraph(GraphState)
workflow.add_node("docs_retrieval", docs_retrieval)
workflow.add_node("relevance_check", relevance_check)
workflow.add_node("search_travily", search_travily)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("hallucination_check", hallucination_check)

workflow.set_entry_point("docs_retrieval")
workflow.add_edge("docs_retrieval", "relevance_check")
workflow.add_conditional_edges(
   "relevance_check", 
  relevance_check_v_generate_answer_and_search_travily, 
  {
    "generate": "generate_answer",
    "travily": "search_travily"
  })
workflow.add_edge("search_travily", "relevance_check")
workflow.add_edge("generate_answer", "hallucination_check")
workflow.add_conditional_edges(
  "hallucination_check", 
  hallucination_check_v_generate_answer_and_end, 
  {
    "generate": "generate_answer",
    "end": END
  })

app = workflow.compile()
inputs = {"question": "What are the types of agent memory?"}
# inputs = {"question": "Where does Messi play right now?"}


for output in app.stream(inputs):
    for key, value in output.items():
        pprint(f"Finished running: {key}:")
pprint(value["answer"])
###################################