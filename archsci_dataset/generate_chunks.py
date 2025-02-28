from collections import deque
# from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
import re

# One token is roughly 0.75 words
# 512 tokens = 
CHUNK_SIZE=384
OVERLAP_SMALL=56    # 15%
OVERLAP_MEDIUM=72   # 20%
OVERLAP_LARGE=96    # 25%

# def generate_chunks_char_splitter(text:str) -> list[str]: 
#     text_splitter = CharacterTextSplitter( 
#         separator="", 
#         chunk_size=CHUNK_SIZE, 
#         chunk_overlap=OVERLAP_SMALL, 
#         length_function=len, 
#         is_separator_regex=False,
#     ) 
#     chunk_doc1 = text_splitter.split_text(text)
#     return chunk_doc1

# def generate_chunks_recursive_char_splitter(text:str) -> list[str]: 
#     text_splitter = RecursiveCharacterTextSplitter( 
#         chunk_size=CHUNK_SIZE, 
#         chunk_overlap=OVERLAP_SMALL, 
#         length_function=len, 
#         is_separator_regex=False, 
#         separators=["\n\n", "\n", " ", ""],
#     )
#     chunk_doc1 = text_splitter.split_text(text)
#     return chunk_doc1


def merge_text_into_paragraph(text:list[str])->list[str]:
    merged_paras = [text[0]]
    for para in text[1:]:
        # check whether the paragraphs starts with lowercase 
        if re.match(r'^\s*[a-z]', para):
            # Merge with previous paragraph
            if merged_paras[-1].startswith('#'):
                merged_paras.append(para)
            else: 
                merged_paras[-1] += ' ' + para
        else:
            if re.match(r'^\s*#+', para):
                if merged_paras[-1][-1] != ".": 
                    merged_paras[-1] += ' ' + para
                else: 
                    merged_paras.append(para)
            else: 
                merged_paras.append(para)
    return merged_paras
        
def split_by_paragraphs(text:str) -> list[str]:
    # Split text into paragraphs using 1+ newlines as separators
    raw_paragraphs = re.split(r'\n{2,}', text.strip())

    raw_paragraphs = merge_text_into_paragraph(raw_paragraphs) 
    # Clean up paragraphs: remove internal newlines and trim whitespace
    cleaned_paragraphs = []
    for para in raw_paragraphs:
        # Replace internal newlines with a space and collapse whitespace
        cleaned = re.sub(r'\s+', ' ', para.replace('\n', ' ')).strip()
        if cleaned:  # Ignore empty paragraphs
            cleaned_paragraphs.append(cleaned)
    return cleaned_paragraphs

def split_into_sections(text:str) -> list[str]:
    # Regex pattern to capture sections (including headers and content)
    pattern = re.compile(
        r'(?m)^##?\s+[A-Z].*?(?=\n##?\s+[A-Z]|\Z)',  # Matches headers and their content 
        flags=re.DOTALL
    ) 
    
    # Find all sections 
    sections = pattern.findall(text)

    return sections

def split_into_section_paragraphs(text:str) -> list[dict]:
    sections = split_into_sections(text)
    paras = []
    for section_id, section in enumerate(sections): 
        paragraphs = split_by_paragraphs(section)
        for idx, para in enumerate(paragraphs): 
            current_section = {
                "section": section, 
                "section_id":section_id,
                "paragraph": para,
                "paragraph_id": idx,
                "length": len(para)
            }
            paras.append(current_section)
        
    return paras

def split_by_headings(text:str) -> list[str]:
    # Regex pattern to capture sections (including headers and content)
    pattern = re.compile(
        r'(?m)^#+\s+[A-Z].*?(?=\n#+\s+[A-Z]|\Z)',  # Matches headers and their content 
        flags=re.DOTALL
    ) 
    
    # Find all sections 
    sections = pattern.findall(text)
    return sections

def split_into_sentences(text:str) -> list[str]:
    """
    Split paragraphs into sentences.
    
    Args:
        text (str): Text containing one or more paragraphs
        
    Returns:
        list: A list of sentences
    """
    # Handle common abbreviations to avoid false sentence breaks
    text = text.replace("Mr.", "Mr_DOT_")
    text = text.replace("Mrs.", "Mrs_DOT_")
    text = text.replace("Dr.", "Dr_DOT_")
    text = text.replace("Ms.", "Ms_DOT_")
    text = text.replace("Prof.", "Prof_DOT_")
    text = text.replace("Inc.", "Inc_DOT_")
    text = text.replace("Ltd.", "Ltd_DOT_")
    text = text.replace("i.e.", "i_e_DOT_")
    text = text.replace("e.g.", "e_g_DOT_")
    text = text.replace("vs.", "vs_DOT_")
    text = text.replace("etc.", "etc_DOT_")
    text = text.replace("Ph.D.", "PhD_DOT_")
    
    # Split on sentence boundaries (., !, ?)
    # Look for: period/exclamation/question mark + space + capital letter or end of string
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z]|$)', text)
    
    # Restore abbreviations
    result = []
    for sentence in sentences:
        sentence = sentence.replace("Mr_DOT_", "Mr.")
        sentence = sentence.replace("Mrs_DOT_", "Mrs.")
        sentence = sentence.replace("Dr_DOT_", "Dr.")
        sentence = sentence.replace("Ms_DOT_", "Ms.")
        sentence = sentence.replace("Prof_DOT_", "Prof.")
        sentence = sentence.replace("Inc_DOT_", "Inc.")
        sentence = sentence.replace("Ltd_DOT_", "Ltd.")
        sentence = sentence.replace("i_e_DOT_", "i.e.")
        sentence = sentence.replace("e_g_DOT_", "e.g.")
        sentence = sentence.replace("vs_DOT_", "vs.")
        sentence = sentence.replace("etc_DOT_", "etc.")
        sentence = sentence.replace("PhD_DOT_", "Ph.D.")
        result.append(sentence)
    
    return result

def chunk_paragraphs(paras:list[dict], max_length=384) -> list[str]:
    """
    Split paragraphs into chunks of maximum length while preserving sentence integrity.
    
    Args:
        paras (list): List of paragraph dictionaries with keys 'section_id', 'paragraph_id', 
                     'paragraph', and 'length'
        max_length (int): Maximum length of each chunk
        
    Returns:
        list: A list of text chunks, each below the maximum length
    """
    chunks = []
    current_chunk = ""
    current_length = 0
    
    for idx, para in enumerate(paras):
        print(f"{para['section_id']}, {para['paragraph_id']}, {para['length']}")
        print("-" * 60)
        
        # Check if adding the whole paragraph would exceed the limit
        if current_length + para["length"] > max_length:
            # Need to split paragraph into sentences
            sentences = split_into_sentences(para["paragraph"])
            
            # Process each sentence
            for sentence in sentences:
                sentence_length = len(sentence)
                
                # If current sentence fits in the current chunk
                if current_length + sentence_length + 1 <= max_length:  # +1 for space
                    if current_chunk:  # Add space if chunk isn't empty
                        current_chunk += " "
                        current_length += 1
                    current_chunk += sentence
                    current_length += sentence_length
                else:
                    # Save current chunk if not empty
                    if current_chunk:
                        chunks.append(current_chunk)
                    
                    # Start a new chunk with this sentence if it fits in a single chunk
                    if sentence_length <= max_length:
                        current_chunk = sentence
                        current_length = sentence_length
                    else:
                        # Handle sentences longer than max_length by splitting at word boundaries
                        words = sentence.split()
                        current_chunk = words[0]
                        current_length = len(words[0])
                        
                        for word in words[1:]:
                            if current_length + len(word) + 1 <= max_length:
                                current_chunk += " " + word
                                current_length += len(word) + 1
                            else:
                                chunks.append(current_chunk)
                                current_chunk = word
                                current_length = len(word)
        else:
            # If we already have content in the chunk, add a newline
            if current_chunk:
                current_chunk += "\n"
                current_length += 1
                
            # Add the whole paragraph
            current_chunk += para["paragraph"]
            current_length += para["length"]
        
        # If the chunk is getting close to max_length, save it
        if current_length > max_length * 0.8:
            chunks.append(current_chunk)
            current_chunk = ""
            current_length = 0
    
    # Don't forget to add the last chunk if it has content
    if current_chunk:
        chunks.append(current_chunk)
    
    # Print results
    print(f"Total chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1} length: {len(chunk)}")
        print(chunk)
        print("+" * 60)
    
    return chunks

def count_words(text:str) -> int:
    """Count the number of words in text."""
    return len(text.split())

def chunk_paragraphs_with_overlap(paras, max_words=384, min_overlap_words=56, max_overlap_words=96):
    """ 
    Split paragraphs into chunks of maximum length while preserving complete sentences
    and including overlapping text between adjacent chunks.
    
    Args:
        paras (list): List of paragraph dictionaries with keys 'section_id', 'paragraph_id', 
                     'paragraph', and 'length'
        max_length (int): Maximum length of each chunk
        min_overlap_words (int): Minimum number of words to overlap between chunks
        max_overlap_words (int): Maximum number of words to overlap between chunks
        
    Returns:
        list: A list of text chunks, each below the maximum length with appropriate overlap
    """
    chunks = []
    current_chunk = ""
    sentence_queue = deque() # Use a deque to track sentences in the current chunk
    current_word_count = 0
    
    for idx, para in enumerate(paras):
        print(f"{para['section_id']}, {para['paragraph_id']}, {para['length']}")
        print("-" * 60)
        
        # Split paragraph into sentences
        paragraph_sentences = split_into_sentences(para["paragraph"])

        # Process each sentence
        for sentence in paragraph_sentences:
            sentence_word_count = count_words(sentence)
            space_count = 1 if current_chunk else 0  # Add space if not first sentence
            
            # If adding this sentence doesn't exceed max_words
            if current_word_count + sentence_word_count + space_count <= max_words:
                if current_chunk:  # Add space if not first sentence
                    current_chunk += " "
                    current_word_count += space_count
                current_chunk += sentence
                sentence_queue.append(sentence)
                current_word_count += sentence_word_count
            else:
                # If the current sentence is too long by itself and we have an empty chunk,
                # we need to split the sentence by words
                if current_word_count == 0 and sentence_word_count > max_words:
                    words = sentence.split()
                    current_chunk = " ".join(words[:max_words])
                    chunks.append(current_chunk)
                    
                    # Continue with remaining words in next chunk
                    remaining_words = words[max_words:]
                    current_chunk = " ".join(remaining_words)
                    current_word_count = len(remaining_words)
                    sentence_queue = deque([current_chunk])
                    continue
                
                # Save current chunk
                chunks.append(current_chunk)
                
                # Calculate overlap using complete sentences
                overlap_words = 0
                overlap_sentences = []
                
                # Work backwards through sentences to create overlap
                for s in reversed(sentence_queue):
                    s_word_count = count_words(s)
                    if overlap_words + s_word_count <= max_overlap_words:
                        overlap_sentences.insert(0, s)
                        overlap_words += s_word_count
                    else:
                        # If we haven't met minimum overlap, take this sentence anyway
                        if overlap_words < min_overlap_words:
                            overlap_sentences.insert(0, s)
                            overlap_words += s_word_count
                        break
                
                # Start new chunk with overlapping sentences
                current_chunk = " ".join(overlap_sentences)
                sentence_queue = deque(overlap_sentences)
                current_word_count = overlap_words
                
                # Add the current sentence if it's not already in the overlap
                if sentence not in overlap_sentences:
                    current_chunk += " " + sentence
                    sentence_queue.append(sentence)
                    current_word_count += sentence_word_count + 1  # +1 for space
        
        # Add paragraph separator if not at end of chunk
        if idx < len(paras) - 1:
            current_chunk += "\n"
            sentence_queue.append("\n")
            # Newline doesn't count as a word
    
    # Don't forget to add the last chunk if it has content
    if current_chunk:
        chunks.append(current_chunk)
    
    # Print results
    print(f"Total chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1} words-length: {count_words(chunk)}")
        print(chunk)
        print("+" * 60)
        
        # Show overlap with next chunk if applicable
        if i < len(chunks) - 1:
            overlap = find_sentence_overlap(chunk, chunks[i+1])
            overlap_word_count = len(" ".join(overlap).split())
            print(f"Overlap with next chunk: {overlap_word_count} words")
            print(f"Overlap sentences: {len(overlap)}")
    
    return chunks

def find_sentence_overlap(chunk1, chunk2):
    """
    Find and display the overlapping words between two adjacent chunks.
    
    Args:
        chunk1 (str): The first chunk
        chunk2 (str): The second chunk
        
    Returns:
        list: The overlapping words between the chunks
    """
    # Split into sentences
    sentences1 = split_into_sentences(chunk1)
    sentences2 = split_into_sentences(chunk2)
    
    # Find where the overlap starts
    overlap = []
    max_overlap = min(len(sentences1), len(sentences2))
    
    for i in range(max_overlap):
        # Compare sentences from the end of first chunk with beginning of second chunk
        if sentences1[-(i+1):] == sentences2[:i+1]:
            overlap = sentences2[:i+1]
            break
    
    return overlap