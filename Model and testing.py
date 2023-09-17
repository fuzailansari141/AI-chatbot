from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS


secret_key="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"

loader = CSVLoader(file_path='brainloxinfoclean.csv', encoding="utf-8")
data = loader.load()

embeddings = OpenAIEmbeddings(openai_api_key=secret_key)
vectors = FAISS.from_documents(data, embeddings)
chain = ConversationalRetrievalChain.from_llm(llm=ChatOpenAI(temperature=0.5, model_name='gpt-3.5-turbo', openai_api_key=secret_key), retriever=vectors.as_retriever())
history=[]
while True:
    query = input("Enter Your Query:")
    print(chain({"question": query, "chat_history": history})["answer"])

