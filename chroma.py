from optimum.rbln import RBLNXLMRobertaModel
from transformers import AutoTokenizer
import os
from typing import List
import torch

from langchain.vectorstores import Chroma
from langchain_docling.loader import DoclingLoader

from langchain_text_splitters import MarkdownHeaderTextSplitter


class MyEmbeddings:
	def __init__(self, model_id="BAAI/bge-m3"):
			self.model = RBLNXLMRobertaModel.from_pretrained(
					model_id=os.path.basename(model_id),
					export=False,
			)
			self.tokenizer = AutoTokenizer.from_pretrained(model_id)

	def embed_documents(self, texts: List[str]) -> List[List[float]]:
			embeddings = []
			for text in texts:
					embedding = self._get_embedding(text)
					embeddings.append(embedding.tolist())
			return embeddings

	def embed_query(self, query: str) -> List[float]:
			embedding = self._get_embedding(query)
			return embedding.tolist()

	def _get_embedding(self, text: str) -> torch.Tensor:
			inputs = self.tokenizer(text, padding="max_length", return_tensors="pt", max_length=8192)
			with torch.no_grad():
					output = self.model(inputs.input_ids, inputs.attention_mask)
			print("model output:", output)
			embedding = torch.nn.functional.normalize(output[0][:, 0], dim=-1)
			return embedding.squeeze()

	def __call__(self, texts):
			if isinstance(texts, str):
					return self.embed_query(texts)
			return self.embed_documents(texts)
      
class ChromaDB:
    def __init__(self, splits, embedding):
        self.store = Chroma.from_documents(
            documents=splits,
            embedding=embedding,
            persist_directory="./chroma_data",
            collection_name="docling_collection"
        )
        self.store.persist()


if __name__ == "__main__":
	########### embedding, ChromaDB S ###########
	print("docling start....")
	loader = DoclingLoader(file_path="./output.md")
	docs = loader.load()
	print("docling loaded....", docs)
	print("md loaded....")

	splitter = MarkdownHeaderTextSplitter(
		headers_to_split_on=[
			("#", "Header_1"),
			("##", "Header_2"),
			("###", "Header_3"),
		]
	)

	splits = [split for doc in docs for split in splitter.split_text(doc.page_content)]
	print("md splits....")
	embedding = MyEmbeddings(model_id="BAAI/bge-m3")
	chroma_db = ChromaDB(splits, embedding)
	print("embedding....")
	########### embedding, ChromaDB e ###########

