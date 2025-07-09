from langchain_docling.loader import DoclingLoader, ExportType


print("docling convert start....")
# source = "./docling.pdf"
source = "./docling.pdf"
loader = DoclingLoader(file_path=source, export_type=ExportType.MARKDOWN)

docs = loader.load()

print("docs....", docs)


with open("./output.md", "w") as f:
	for doc in docs:
		# Document 객체의 page_content를 문자열로 변환하여 쓰기
		f.write(doc.page_content)
		# f.write("\n\n")  # 문서 간 구분을 위한 줄바꿈 추가
    
	f.close()
print("docling convert end....")