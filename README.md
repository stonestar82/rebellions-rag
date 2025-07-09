# Rebellions RAG

- Rebellions NPU RAG(Retrieval-Augmented Generation) 테스트
- LLM : EVE-Korean-Instruct-10.8B-v1.0
- Embedding : bge-m3
- Reranker : ko-reranker
- RangChain

## 📋 실행 순서

다음 순서대로 스크립트를 실행하세요:

### 1. 모델 컴파일 단계

```bash
python eeve_compile.py
python bge_compile.py
```

### 2. 문서 처리 단계

```bash
python pdf2md.py
```

### 3. 벡터 데이터베이스 설정

```bash
python chroma.py
```

### 4. 재순위화 모델 컴파일

```bash
python reranker_compile.py
```

### 5. RAG 실행

```bash
python rag.py
```

## 📁 프로젝트 구조

- `eeve_compile.py` - EEVE 모델 컴파일
- `bge_compile.py` - BGE 모델 컴파일
- `pdf2md.py` - PDF 문서를 Markdown으로 변환
- `chroma.py` - Chroma 벡터 데이터베이스 설정
- `reranker_compile.py` - 재순위화 모델 컴파일
- `rag.py` - 메인 RAG 시스템

## 🚀 시작하기

1. 필요한 의존성을 설치하세요
2. 위의 실행 순서에 따라 스크립트를 순차적으로 실행하세요
3. 각 단계가 성공적으로 완료된 후 다음 단계로 진행하세요

## 📝 참고사항

- 각 스크립트는 이전 단계의 완료를 전제로 합니다
- 오류가 발생하면 이전 단계를 다시 확인하세요
- `docling.pdf` 파일이 프로젝트에 포함되어 있습니다
