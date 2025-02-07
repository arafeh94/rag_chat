import os
import uuid
from typing import TypedDict, List, Annotated, Sequence

import langchain_community.document_loaders as loaders
import langchain_text_splitters as text_splits
from langchain_chroma import Chroma
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START
from langgraph.graph import StateGraph, add_messages
from langgraph.graph.state import CompiledStateGraph


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    question: str
    context: List[Document]
    source: List[str]


class Chat:
    def __init__(self, id, graph: CompiledStateGraph):
        self.id = id
        self.graph = graph

    def answer(self, question, source=None) -> BaseMessage:
        config = {"configurable": {"thread_id": self.id}}
        response = self.graph.invoke({"question": question, "source": source}, config)
        return response["messages"][-1]


class ChatService:
    def __init__(self, vector_store, llm):
        self.vector_store = vector_store
        self.llm = llm

    def embed_pdf(self, path):
        ts = text_splits.RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.vector_store.add_documents(loaders.PyPDFLoader(path).load_and_split(ts))

    def _retrieve(self, state: State):
        filter = {'source': {'$in': state['source']}} if state['source'] else None
        results = self.vector_store.similarity_search_with_score(
            state['question'],
            filter=filter
        )
        context = [d for d, s in results]
        return {"context": context}

    def _prompt(self, state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        history = "\n\n".join(self._role(msg) + ":" + msg.content for msg in state["messages"])
        context = "Document Information Start:\n" + docs_content + "\nDocument Information  End\n\n"
        context += "Chat History Start:\n" + history + "\nChat History End\n\n"
        context = "Using the following information:\n" + context + "\nAnswer the human question"
        prompt = """
        You are a professional AI assistant (no emojis) with retrieval-augmented generation capabilities. 
        When given a user query, first retrieve the most relevant information.
        Information can be collected from the provided knowledge base, documents and chat history. 
        Then, generate a well-structured response that accurately integrates the retrieved information. 
        If the retrieved content is insufficient, provide a best-effort response while indicating uncertainty."""
        context = prompt + "\n" + context
        return {"context": context}

    def _prompt2(self, state: State):
        messages = []
        prompt = """
        You are a professional AI assistant (no emojis) with retrieval-augmented generation capabilities. 
        When given a user query, first retrieve the most relevant information.
        Information can be collected from the provided knowledge base, documents and chat history. 
        Then, generate a well-structured response that accurately integrates the retrieved information. 
        If the retrieved content is insufficient, provide a best-effort response while indicating uncertainty."""
        messages.append(('system', prompt))
        for doc in state['context']:
            messages.append(('system', doc.page_content))
        for message in state['messages']:
            messages.append((self._role(message), message.content))
        return {"context": messages}

    def _generate(self, state: State):
        question = state["question"]
        messages = [("system", state['context']), ('human', state['question'])]
        response = self.llm.invoke(messages)
        return {"messages": [HumanMessage(question), response]}

    def _generate2(self, state: State):
        question = state["question"]
        messages = [*state['context'], ('human', state['question'])]
        response = self.llm.invoke(messages)
        return {"messages": [HumanMessage(question), response]}

    def _role(self, message: BaseMessage):
        return 'human' if message.__class__.__name__ == 'HumanMessage' else 'assistant'

    def new_chat(self, v=1):
        sequence = [self._retrieve]
        sequence += [self._prompt, self._generate] if v == 1 else [self._prompt2, self._generate2]
        token = uuid.uuid4()
        graph_builder = StateGraph(State).add_sequence(sequence)
        graph_builder.add_edge(START, "_retrieve")
        graph = graph_builder.compile(checkpointer=MemorySaver())
        return Chat(token, graph)

    def dummy(self):
        self.embed_pdf('./files/1.pdf')
        self.embed_pdf('./files/2.pdf')
        self.embed_pdf('./files/3.pdf')

    @staticmethod
    def create():
        return ChatService.create_openai()

    @staticmethod
    def create_openai(model_name='gpt-4o', temperature=0):
        embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
        llm = ChatOpenAI(name=model_name, temperature=temperature, openai_api_key=os.environ['OPENAI_API_KEY'])

        vector_store = Chroma(embedding_function=embeddings, persist_directory='./vs2')
        return ChatService(vector_store, llm)

    @staticmethod
    def create_ollama(model_name="gemma2:2b", temperature=0):
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        llm = ChatOllama(model=model_name, temperature=temperature)

        vector_store = Chroma(embedding_function=embeddings, persist_directory='./vs2')
        return ChatService(vector_store, llm)
