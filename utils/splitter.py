from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " "]
    )
    
    # Ici, on split les documents (objet langchain.schema.Document)
    return splitter.split_documents(documents)
