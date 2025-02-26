from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# One token is roughly 0.75 words
# 512 tokens = 
CHUNK_SIZE=384
OVERLAP_SMALL=56    # 15%
OVERLAP_MEDIUM=72   # 20%
OVERLAP_LARGE=96    # 25%

def generate_chunks_char_splitter(text:str) -> list[str]: 
    text_splitter = CharacterTextSplitter( 
        separator="", 
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=OVERLAP_SMALL, 
        length_function=len, 
        is_separator_regex=False,
    ) 
    chunk_doc1 = text_splitter.split_text(text)
    return chunk_doc1


def generate_chunks_recursive_char_splitter(text:str) -> list[str]: 
    text_splitter = RecursiveCharacterTextSplitter( 
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=OVERLAP_SMALL, 
        length_function=len, 
        is_separator_regex=False, 
        separators=["\n\n", "\n", " ", ""],
    )
    chunk_doc1 = text_splitter.split_text(text)
    return chunk_doc1

