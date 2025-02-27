
import re
from collections import deque

def preprocess_footnote_v2(text: str) -> str:
    """
    Detects and removes footnotes from text based on standard footnote patterns.
    
    Args:
        text (str): The input text potentially containing footnotes
        
    Returns:
        str: Text with footnotes removed, or the original text if no footnotes detected
    """
    # More comprehensive footnote pattern that captures various footnote formats
    # 1. Standard numbered footnotes (e.g., "1 This is a footnote.")
    # 2. Superscript footnotes (e.g., "[1] This is a footnote.")
    # 3. Asterisk footnotes (e.g., "* This is a footnote.")
    footnote_patterns = [
        # Pattern 1: Number at start of line followed by text ending with period
        r'^\d+\s+[A-Z].*\.$',
        # Pattern 2: Number in brackets/parentheses at start of line followed by text
        r'^\[\d+\]\s+.*\.$',
        r'^\(\d+\)\s+.*\.$',
        # Pattern 3: Asterisk or other common footnote markers
        r'^\*\s+.*\.$',
        r'^†\s+.*\.$'
    ]
    
    # Try each pattern with DOTALL flag to handle multi-line footnotes
    for pattern in footnote_patterns:
        regex = re.compile(pattern, re.DOTALL)
        match = regex.search(text)
        if match:
            # Return empty string to remove the footnote
            return ""
    
    # Additional check for numbered text that spans multiple paragraphs
    # (common in legal or academic footnotes)
    if re.match(r'^\d+\s+', text) and text.strip().endswith('.'):
        return ""
    
    # If no patterns match, return the original text
    return text

def preprocess_footnote(text:str) -> str:
    # Regex pattern with DOTALL flag to include newlines 
    footnote_pattern = re.compile(r'^\d+\s.*\.', re.DOTALL)
    match = footnote_pattern.search(text)
    if match:
        #print(match.group())
        return ""
    else:
        #print("no match")
        return text

def preprocess_footnote_intext_replace(text:str, matched_list:list) -> str:
    """
    Removes footnote reference numbers from text while preserving DOIs and other numeric identifiers.
    
    Args:
        text (str): The input text containing footnote references
        matched_list (list): List of footnote numbers to remove (as strings)
        
    Returns:
        str: Text with footnote references removed
    """
    # Validate the matched_list to ensure it contains only valid reference numbers
    valid_references = []

    for ref in matched_list:
        # Ensure each reference is a string and consists of digits
        if isinstance(ref, str) and ref.isdigit():
            valid_references.append(re.escape(ref))
        elif isinstance(ref, int):
            valid_references.append(re.escape(str(ref)))
    
    if not valid_references:
        return text  # Return original text if no valid references
    
    # Regex pattern to match footnote numbers
    footnote_pattern = re.compile( 
        r''' 
        (?<![dD][oO][iI])# Negative lookbehind for "doi" (case-insensitive) 
        (?<!10\.)        # Negative lookbehind for "10." (common DOI prefix)
        (?<=             # Positive lookbehind to ensure the number is preceded by: 
          [a-zA-Z,.;’”]  # A letter or common punctuation (e.g., comma, period, quote)
        ) 
        ({})             # Match specific numbers (formatted into the pattern)
        (?=              # Positive lookahead to ensure the number is followed by: 
          \W|$           # A non-word character (e.g., space, semicolon) or end of string
        ) 
        '''.format('|'.join(valid_references)),
        flags=re.VERBOSE
    ) 
    
    # Remove the footnote numbers from the text
    clean_text = footnote_pattern.sub('', text)

    # Remove any double spaces that might have been created
    clean_text = re.sub(r'\s{2,}', ' ', clean_text)
    
    # Fix spaces before punctuation
    clean_text = re.sub(r'\s+([,.;:?!])', r'\1', clean_text)
    
    return clean_text

def preprocess_footnote_intext(text:str) -> list:
    # Regex pattern to match footnote numbers

    footnote_pattern = re.compile( 
        r''' 
        (?<![dD][oO][iI])# Negative lookbehind for "doi" (case-insensitive) 
        (?<!10\.)        # Negative lookbehind for "10." (common DOI prefix)
        (?<=             # Positive lookbehind to ensure the number is preceded by: 
          [a-zA-Z,.;’”]  # A letter or common punctuation (e.g., comma, period, quote)
        ) 
        (\d+)            # Match the footnote number (one or more digits) 
        (?=              # Positive lookahead to ensure the number is followed by: 
          \W|$           # A non-word character (e.g., space, semicolon) or end of string
        ) 
        ''', 
        flags=re.VERBOSE
    ) 
    
    # Extract all footnote numbers 
    matches = footnote_pattern.findall(text)
    return matches 

with open("./data/journals/00002.md") as fp:
    content = fp.readlines()

text = ""
footnotes_identified = []
last_footnotes = ''
for line in content:
    tmp_str = preprocess_footnote_v2(line)
    clean_text = ''
    if tmp_str != "": 
        matches_footnote = preprocess_footnote_intext(line)
        print("==============")
        print(matches_footnote)
        print("===== 1 =========")
        print(footnotes_identified)
        if len(footnotes_identified) > 0: 
            last_footnotes = footnotes_identified[-1]
        print("===== 2 =========")
        print(last_footnotes)
        if len(matches_footnote) > 0:
            if matches_footnote[0] > last_footnotes: 
                clean_text = preprocess_footnote_intext_replace(line, matches_footnote)
                footnotes_identified.extend(matches_footnote)
            
        if len(clean_text) > 0: 
            text += clean_text
        else:
            text += line

print("=========================")
print(text)
print("----------- 1. preprocess footnote ============")
# 1. preprocess_footnote
print(preprocess_footnote(text))

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
            print("aDDDDD")
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


def chunksize(text: list[dict]) -> list[str]:
    chunk_docs=[]

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


paras = split_into_section_paragraphs(text)
#chunks = chunk_paragraphs(paras)
chunks_overlap = chunk_paragraphs_with_overlap(paras)
# current_length = 0
# chunks = []
# count_chunks = 0
# current_chunk = ""
# for idx,para in enumerate(paras):
#     print(para["section_id"], para["paragraph_id"], para["length"])
#     print("-" * 60) 
#     if current_length + para["length"] > 384: 
#         # need to split para 
#         sentences = split_into_sentences(para["paragraph"])
#         leftover_sentences = [] 
#         for sentence in sentences: 
#             if len(sentence) + current_length < 384: 
#                 current_chunk += " " + sentence 
#                 current_length += len(sentence) 
#             else: 
#                 leftover_sentences.append(sentence)
#         chunks.append(current_chunk)
#         current_chunk = ""
#         if len(leftover_sentences) > 0:
#             current_chunk += ' '.join(leftover_sentences)
#             current_length = len(current_chunk)
#     else:
#         current_chunk += "\n" + para["paragraph"]
#         current_length = len(current_chunk)

# print(len(chunks))
# for chunk in chunks:
#     print(len(chunk),chunk)                  
#     print("+" * 60)                   




#split_by_headings(text)


exit()

from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    separator="",
    chunk_size=384,
    chunk_overlap=56,
    length_function=len,
    is_separator_regex=False,
)
chunk_doc1 = text_splitter.split_text(text)
for index, elem in enumerate(chunk_doc1):
    print(index, elem)

from langchain_text_splitters import RecursiveCharacterTextSplitter

print(text)
text_splitter2 = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=1,
    length_function=len,
    is_separator_regex=False,
    separators=["\n\n", "\n", " ", ""],
)

chunk_doc2 = text_splitter2.split_text(text)

for i , _ in enumerate(chunk_doc2):
    print(f"chunk #{i} , size:{len(chunk_doc2[i])}")
    print(chunk_doc2[i])
    print("-----")
print("sssssemantic lllllll")
from sentence_transformers import SentenceTransformer  # noqa: E402
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import pandas as pd


embedding_model = "Lajavaness/bilingual-embedding-large"

my_embedding_model = SentenceTransformer(embedding_model, trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(embedding_model, trust_remote_code=True)


def plot_chunk(chunk_doc, embedding_name):
    tokenizer = AutoTokenizer.from_pretrained(embedding_model, trust_remote_code=True)
    length = [len(tokenizer.encode(doc.page_content)) for doc in chunk_doc2]
    fig = pd.Series(length).hist()
    plt.show()


content = text

split_char = [".", "?", "!"]

import re  # noqa: E402

sentences_list = re.split(r"(?<=[.?!])\s+", content)
print(sentences_list)
print(len(sentences_list))

sentences = [{"sentence": x, "index": i} for i, x in enumerate(sentences_list)]
print("Sssssentences")
print(sentences)
exit()


def combine_sentences(sentences, buffer_size=1):
    combine_sentences = [
        " ".join(
            sentences[j]["sentence"]
            for j in range(
                max(i - buffer_size, 0), min(i + buffer_size + 1, len(sentences))
            )
        )
        for i in range(len(sentences))
    ]

    for i, combine_sentence in enumerate(combine_sentences):
        sentences[i]["combined_sentence"] = combine_sentence

    return sentences


sentences = combine_sentences(sentences=sentences)

print(sentences)
print(len(sentences))
embeddings = my_embedding_model.encode(
    sentences=[x["combined_sentence"] for x in sentences]
)
for i, elem in enumerate(sentences):
    print(elem)
    elem["combined_sentence_embedding"] = embeddings[i]

print("combinedddddd")
for elem in sentences:
    print(elem)
print("combinedddddd___________")

import numpy as np


def cosine_similarity(vec1:np.ndarray, vec2:np.ndarray) -> float:
    # Calculate the similarity between two vectors
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


def calculate_cosine_distances(sentences):
    distances = []
    for i in range(len(sentences) - 1):
        embedding_current = sentences[i]["combined_sentence_embedding"]
        embedding_next = sentences[i + 1]["combined_sentence_embedding"]

        # Calculate cosine similarity
        similarity = cosine_similarity(embedding_current, embedding_next)

        distance = 1 - similarity
        distances.append(distance)

        # Store into the sentences
        sentences[i]["distance_to_next"] = distance
    return distances, sentences


distances, sentences = calculate_cosine_distances(sentences=sentences)

print(sentences[-2]["distance_to_next"])

y_upper_bound = 0.25
plt.ylim(0, y_upper_bound)
plt.xlim(0, len(distances))

# percentile()
brekpoint_percentile_threshold = 95
brekpoint_distance_threshold = np.percentile(distances, brekpoint_percentile_threshold)

plt.axhline(y=brekpoint_distance_threshold, color="r", linestyle="-")
num_distances_above_threshold = len(
    [x for x in distances if x > brekpoint_distance_threshold]
)
plt.text(
    x=(len(distances) * 0.01),
    y=y_upper_bound / 50,
    s=f"{num_distances_above_threshold +1} chunks",
)

indices_above_thresh = [
    i for i, x in enumerate(distances) if x > brekpoint_distance_threshold
]

colors = ["b", "g", "r", "c", "m", "y", "k"]

for i, breakpoint_index in enumerate(indices_above_thresh):
    start_index = 0 if i == 0 else indices_above_thresh[i - 1]
    end_index = (
        breakpoint_index if i <= len(indices_above_thresh) - 1 else len(distances)
    )

    plt.axvspan(start_index, end_index, facecolor=colors[i % len(colors)], alpha=0.25)
    plt.text(
        x=np.average([start_index, end_index]),
        y=brekpoint_distance_threshold + (y_upper_bound) / 20,
        s=f"Chunk #{i}",
        horizontalalignment="center",
        rotation="vertical",
    )

    if indices_above_thresh:
        last_breakpoint = indices_above_thresh[-1]
        if last_breakpoint < len(distances):
            plt.axvspan(
                last_breakpoint,
                len(distances),
                facecolor=colors[len(indices_above_thresh) % len(colors)],
                alpha=0.25,
            )
            plt.text(
                x=np.average([last_breakpoint, len(distances)]),
                y=brekpoint_distance_threshold + (y_upper_bound) / 20,
                s=f"Chunk #{i+1}",
                horizontalalignment="center",
                rotation="vertical",
            )

plt.plot(distances)
plt.show()

start_index = 0
chunks = []

for index in indices_above_thresh:
    end_index = index
    group = sentences[start_index : end_index + 1]
    combined_text = " ".join([d["sentence"] for d in group])
    chunks.append(combined_text)

    start_index = index + 1


if start_index < len(sentences):
    combine_sentences = " ".join([d["sentence"] for d in sentences[start_index:]])
    chunks.append(combined_text)

for i, chunk in enumerate(chunks[:1]):
    buffer = 200
    print(f"Chunk #{i}")
    print(chunk[:buffer].strip())
    print("...")
    print(chunk[-buffer:].strip())
    print("\n")


for i, chunk in enumerate(chunks):
    print(i, chunk)

print(chunks)
