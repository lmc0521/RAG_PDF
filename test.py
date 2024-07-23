from langchain.chains.llm import LLMChain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import PyMuPDFLoader
#使用文件載入器
loader = PyMuPDFLoader("Virtual_characters.pdf")
PDF_data = loader.load()
# print('文件載入:',PDF_data)

from langchain_text_splitters import RecursiveCharacterTextSplitter
#使用text_splitters分割文件
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=5)
all_splits = text_splitter.split_documents(PDF_data)
# print('分割結果',all_splits)

from langchain_huggingface import HuggingFaceEmbeddings
#載入embedding model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cpu'}
embedding = HuggingFaceEmbeddings(model_name=model_name,
                                  model_kwargs=model_kwargs)

from langchain_community.vectorstores import Chroma
#將向量化結果存入vector DB
persist_directory = 'db'
vectordb = Chroma.from_documents(documents=all_splits, embedding=embedding, persist_directory=persist_directory)

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp
# model_path = "llama-2-7b.Q4_K_M.gguf"
model_path='llama-2-13b-chat.Q5_K_M.gguf'
# model_path='llama-2-7b.Q8_0.gguf'
#使用LlamaCpp載入Higging Face Hub下載的llama-2-7b.Q4_K_M.gguf模型
llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=100,
    n_batch=512,
    n_ctx=2048,
    f16_kv=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=True,
)
# llm("What is Taiwan known for?")

from langchain.chains.prompt_selector import ConditionalPromptSelector
from langchain.prompts import PromptTemplate

DEFAULT_LLAMA_SEARCH_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""<<SYS>>
    You are a helpful assistant eager to assist with providing better Google search results.
    <</SYS>>

    [INST] Provide an answer to the following question in 150 words. Ensure that the answer is informative, \
            relevant, and concise:
            {question}
    [/INST]""",
)

DEFAULT_SEARCH_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are a helpful assistant eager to assist with providing better Google search results. \
        Provide an answer to the following question in about 150 words. Ensure that the answer is informative, \
        relevant, and concise: \
        {question}""",
)

QUESTION_PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=DEFAULT_SEARCH_PROMPT,
    conditionals=[(lambda llm: isinstance(llm, LlamaCpp), DEFAULT_LLAMA_SEARCH_PROMPT)],
)

prompt = QUESTION_PROMPT_SELECTOR.get_prompt(llm)
# # print(prompt)
# # llm_chain = LLMChain(prompt=prompt, llm=llm)
llm_chain=prompt | llm
question = "What is Taiwan known for?"
# llm_chain.invoke({"question": question})

retriever = vectordb.as_retriever()

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    verbose=True
)
#
query = "Tell me about Alison Hawk's career and age"
qa.invoke(query)