from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from rag_pipeline.rag import rag_embed

pipeline = rag_embed()
vectorstore = pipeline["vectorstore"]
model = pipeline["model"]

# 5 — retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

results = retriever.invoke("What helped Revathi to claim her plants...")
# for doc in results:
#     print(doc.page_content)
#     print("---")

# 6 — prompt
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question using only the context provided below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
""")


# 7 — format retrieved docs into a single string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# 8 — chain
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# 9 — run
question = "What helped Revathi to claim her plants - her belief in magic or the belief in her convictions? Explain your choice."

answer = chain.invoke(question)
print(answer)
