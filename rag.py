# optimum.rbln에서 RBLN 기반 Llama, XLMRoberta 모델 임포트
from optimum.rbln import RBLNLlamaForCausalLM, RBLNXLMRobertaModel
# transformers에서 토크나이저 임포트
from transformers import AutoTokenizer
import os
from langchain.prompts import PromptTemplate
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.language_models import BaseLLM
from langchain.chains import LLMChain
from langchain_core.runnables import RunnablePassthrough
from typing import List, Dict, Any, Union, Optional
import torch
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.outputs import LLMResult, Generation

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_qdrant import RetrievalMode
from langchain_qdrant import QdrantVectorStore

from langchain_docling.loader import DoclingLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter

from langchain.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from optimum.rbln import RBLNXLMRobertaForSequenceClassification

from langchain_core.runnables import RunnableLambda

# 커스텀 임베딩 클래스: BAAI/bge-m3 모델을 사용해 문서 임베딩 생성
# CustomEmbeddings 클래스에서 embed_documents, embed_query 구현 이유
# 1. LangChain Embeddings 인터페이스 요구사항
# CustomEmbeddings가 Embeddings를 상속받을 때, 반드시 구현해야 하는 추상 메서드들.
# LangChain 프레임워크가 임베딩을 호출할 때 내부적으로 embed_documents, embed_query 메서드를 사용.
# 2. LangChain과 외부 임베딩 모델 연결
# BAAI/bge-m3 모델은 LangChain과 직접 호환되지 않음.
# embed_documents, embed_query 메서드를 구현함으로써, BAAI/bge-m3 모델을 LangChain의 Embeddings 인터페이스에 맞게 래핑(wrap).
# 이렇게 호출하면 벡터스토어에서 임베딩 생성 시 해당 메서드들이 호출됨.
# vector_store = Chroma(
#     collection_name="docling_collection",
#     embedding_function=CustomEmbeddings(model_id="BAAI/bge-m3"),  # ← 여기서 embed_documents/embed_query가 호출됨
#     persist_directory="./chroma_data"
# )
class CustomEmbeddings(Embeddings):
    def __init__(self, model_id="BAAI/bge-m3"):
        self.model = RBLNXLMRobertaModel.from_pretrained(
            model_id=os.path.basename(model_id),
            export=False,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # 문서 리스트를 임베딩 벡터 리스트로 변환 (벡터스토어 저장용)
        embeddings = []
        for text in texts:
            embedding = self._get_embedding(text)
            embeddings.append(embedding.tolist())
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        # 쿼리 문자열을 임베딩 벡터로 변환 (검색용)
        embedding = self._get_embedding(query)
        return embedding.tolist()

    def _get_embedding(self, text: str) -> torch.Tensor:
        # 입력 텍스트를 토크나이즈 후 임베딩 추출
        # 1. 텍스트를 토크나이즈
        inputs = self.tokenizer(text, padding="max_length", return_tensors="pt", max_length=8192)
        with torch.no_grad():
            # 2. BAAI/bge-m3 모델로 임베딩 생성
            output = self.model(inputs.input_ids, inputs.attention_mask)
        print("model output:", output)
        if output is None or len(output) == 0:
            raise ValueError("Model output is None or empty")
        # 3. CLS 토큰([0] 위치)의 임베딩을 추출하고 정규화
        embedding = torch.nn.functional.normalize(output[0][:, 0], dim=-1)
        return embedding.squeeze()

# 커스텀 LLM 래퍼 클래스: EEVE Llama 모델을 래핑해 LangChain LLM 인터페이스 제공
# CustomLLM 클래스에서 _generate 구현 이유
# 1. LangChain LLM 인터페이스 요구사항
# CustomLLM이 BaseLLM을 상속받을 때, 반드시 구현해야 하는 추상 메서드.
# LangChain 프레임워크가 LLM을 호출할 때 내부적으로 _generate 메서드를 사용.
# 2. LangChain과 외부 모델 연결
# EEVE Llama 모델은 LangChain과 직접 호환되지 않음.
# _generate 메서드를 구현함으로써, EEVE 모델을 LangChain의 LLM 인터페이스에 맞게 래핑(wrap).
# 이렇게 호출하면 rag_chain에서 eeve_llm 호출 시 _generate 메서드가 호출됨.
# rag_chain = (
#    ...
#    | prompt
#    | eeve_llm  # ← 여기서 _generate가 호출됨
#    | StrOutputParser()
#)

class CustomLLM(BaseLLM):
    def __init__(self, model, tokenizer):
        super().__init__()
        self._model = model
        self._tokenizer = tokenizer

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs) -> LLMResult:
        generations = []
        for prompt in prompts:
            # 입력 프롬프트를 토크나이즈 후 LLM으로 생성
            # 1. 프롬프트를 토크나이즈
            inputs = self._tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            # 2. EEVE 모델로 텍스트 생성
            outputs = self._model.generate(**inputs, max_new_tokens=2000, do_sample=True, temperature=0.1, top_k=3, top_p=0.3, repetition_penalty=1.1)
            # 3. 생성된 텍스트를 디코드
            text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            last_line = text.strip().split('\n')[-1]
            # 4. LangChain 형식으로 결과 반환
            generations.append([Generation(text=last_line)])
        return LLMResult(generations=generations)

    @property
    def _llm_type(self) -> str:
        return "custom_llm"

########### EEVE S ###########
# EEVE-Korean-Instruct-10.8B-v1.0 모델 및 토크나이저 로드
model_id = "yanolja/EEVE-Korean-Instruct-10.8B-v1.0"
model = RBLNLlamaForCausalLM.from_pretrained(
    model_id=os.path.basename(model_id),
    export=False,
)
tokenizer = AutoTokenizer.from_pretrained(model_id, model_max_length=512)
tokenizer.pad_token = tokenizer.eos_token

eeve_llm = CustomLLM(model, tokenizer)
########### EEVE E ###########

# 시스템 프롬프트 템플릿 정의 (항상 한국어로 답변)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI Assistant Your name is 'Marcus bot'. You are a smart and humorous and intellectual. YOU ALWAYS ANSWER IN KOREAN. Use the following context to answer the user's question: {context}",
        ),
        ("human", "{question}"),
    ]
)

## 벡터스토어 및 리트리버 준비

# 기존 Chroma 벡터 DB 로드 (docling_collection)
print("Loading existing vector store...")
vector_store = Chroma(
    collection_name="docling_collection",
    embedding_function=CustomEmbeddings(model_id="BAAI/bge-m3"),
    persist_directory="./chroma_data"
)

# 벡터스토어에서 k=10개 문서 검색하는 리트리버 생성
retriever = vector_store.as_retriever(search_kwargs={"k": 10})

# reranker 모델 및 토크나이저 로드 (ko-reranker)
reranker_model = RBLNXLMRobertaForSequenceClassification.from_pretrained(
    model_id="ko-reranker",  # ko-reranker 디렉토리 사용
    export=False,
)
reranker_tokenizer = AutoTokenizer.from_pretrained("ko-reranker")

# reranker: 쿼리-문서 쌍별로 점수 예측 후 상위 N개만 반환
# (ko-reranker는 batch 입력 미지원, for 루프 사용)
def rerank(query, docs, top_n=5):
    pairs = [(query, doc.page_content) for doc in docs]
    scores = []
    for q, d in pairs:
        inputs = reranker_tokenizer(
            q, d,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = reranker_model(**inputs)
            score = outputs.logits.squeeze().item()
        scores.append(score)
    reranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in reranked[:top_n]]

# 문서 리스트를 문자열로 변환 (LLM 입력용)
def format_docs(docs):
    print("docs count:", len(docs))
    return "\n\n".join(doc.page_content for doc in docs)

# 리트리버 결과(docs)와 쿼리(question)를 받아 rerank 후 문자열로 변환
def rerank_and_format_docs(inputs):
    # dict 또는 Namespace/dotdict 등 다양한 타입 지원
    try:
        docs = inputs["docs"]
        question = inputs["question"]
    except Exception:
        docs = getattr(inputs, "docs", None)
        question = getattr(inputs, "question", None)
    reranked_docs = rerank(question, docs, top_n=5)
    return format_docs(reranked_docs)

# rag_chain: 전체 RAG 파이프라인 체인 구성
# 1. 입력에서 question만 추출해 retriever로 문서 검색
# 2. 검색 결과와 question을 dict로 묶어 reranker+format_docs에 전달
# 3. context/question을 prompt에 전달 → LLM → 출력 파싱
rag_chain = (
    RunnableLambda(lambda x: {
        "docs": retriever.invoke(
            x["question"] if isinstance(x, dict) and "question" in x else getattr(x, "question", "")
        ),
        "question": x["question"] if isinstance(x, dict) and "question" in x else getattr(x, "question", "")
    })
    | RunnableLambda(lambda x: {"context": rerank_and_format_docs(x), "question": x["question"] if isinstance(x, dict) and "question" in x else getattr(x, "question", "")})
    | prompt
    | eeve_llm
    | StrOutputParser()
)

# 예시 실행: "Docling이 뭐야?" 질문에 대해 전체 RAG+reranker 파이프라인 동작
result = rag_chain.invoke({"question": "Docling이 뭐야?"})
print(result)
