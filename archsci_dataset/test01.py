import re
from archsci_dataset.preprocess_footnotes import preprocess_footnote_v1, preprocess_footnote_intext, preprocess_footnote_intext_replace, preprocess_footnote
from archsci_dataset.similarity import *
from archsci_dataset.generate_chunks import split_into_section_paragraphs, chunk_paragraphs_with_overlap
from collections import Counter
import math

def main():
    with open("./data/journals/00002.md") as fp:
        content = fp.readlines()

    text = ""
    footnotes_identified = []
    last_footnotes = ''
    for line in content:
        tmp_str = preprocess_footnote_v1(line)
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

    paras = split_into_section_paragraphs(text) 
    #chunks = chunk_paragraphs(paras) 
    
    chunks_overlap = chunk_paragraphs_with_overlap(paras)


def chunksize(text: list[dict]) -> list[str]:
    chunk_docs=[]

def preprocess_sentence_words(sentence):
    """Basic preprocessing for similarity calculations."""
    # Convert to lowercase and remove punctuation
    sentence = re.sub(r'[^\w\s]', '', sentence.lower())
    
    # Split into words
    words = sentence.split()
    
    # Remove common stopwords
    stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", 
            "for", "with", "by", "of", "is", "are", "was", "were", "be", 
            "this", "that", "these", "those", "it", "they", "he", "she"}
    words = [word for word in words if word not in stopwords]
    
    return words

def jaccard_similarity(sentence1, sentence2):
    """Calculate Jaccard similarity between two sentences."""
    words1 = set(preprocess_sentence_words(sentence1))
    words2 = set(preprocess_sentence_words(sentence2))
    
    # Calculate Jaccard similarity
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    if union == 0:  # Handle edge case of empty sets
        return 0
    
    return intersection / union
    
def cosine_similarity(sentence1, sentence2):
    """Calculate cosine similarity between two sentences."""
    words1 = preprocess_sentence_words(sentence1)
    words2 = preprocess_sentence_words(sentence2)
    
    # Count term frequencies
    word_freq1 = Counter(words1)
    word_freq2 = Counter(words2)
    
    # Get all unique words
    all_words = set(word_freq1.keys()).union(set(word_freq2.keys()))
    
    # Calculate dot product and magnitudes
    dot_product = sum(word_freq1.get(word, 0) * word_freq2.get(word, 0) for word in all_words)
    magnitude1 = math.sqrt(sum(word_freq1.get(word, 0) ** 2 for word in all_words))
    magnitude2 = math.sqrt(sum(word_freq2.get(word, 0) ** 2 for word in all_words))
    
    # Handle zero division
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
        
    # Calculate cosine similarity
    return dot_product / (magnitude1 * magnitude2)
    
def calculate_similarity(method, sentence1, sentence2):
    """Calculate similarity using the chosen method."""
    if method.lower() == "cosine":
        return cosine_similarity(sentence1, sentence2)
    else:  # Default to Jaccard
        return jaccard_similarity(sentence1, sentence2)

def chunk_paragraphs_by_similarity(paras, max_words=384, min_overlap_words=56, max_overlap_words=96, 
                                 similarity_threshold=0.2, similarity_method="jaccard"):
    """
    Split paragraphs into chunks based on sentence similarity and word count limits.
    Sentences with higher similarity will be kept together in the same chunk when possible.
    
    Args:
        paras (list): List of paragraph dictionaries with keys 'section_id', 'paragraph_id', 
                     'paragraph', and 'length'
        max_words (int): Maximum number of words in each chunk
        min_overlap_words (int): Minimum number of words to overlap between chunks
        max_overlap_words (int): Maximum number of words to overlap between chunks
        similarity_threshold (float): Threshold for determining sentence similarity (0-1)
        similarity_method (str): Method to calculate similarity - "jaccard" or "cosine"
        
    Returns:
        list: A list of text chunks with semantically similar sentences grouped together
    """
    chunks = []
    current_chunk = ""
    current_word_count = 0
    sentence_queue = deque()
    
    
    # Process all paragraphs
    all_sentences = []
    all_sent_para_mapping = []  # To track which paragraph each sentence came from
    
    # First, collect all sentences from all paragraphs
    for para_idx, para in enumerate(paras):
        paragraph_sentences = split_into_sentences(para["paragraph"])
        for sent in paragraph_sentences:
            all_sentences.append(sent)
            all_sent_para_mapping.append(para_idx)
    
    # Calculate similarity between adjacent sentences
    similarities = []
    for i in range(len(all_sentences) - 1):
        # Only calculate similarity for sentences in the same paragraph
        if all_sent_para_mapping[i] == all_sent_para_mapping[i+1]:
            sim = calculate_similarity(all_sentences[i], all_sentences[i+1])
        else:
            sim = 0  # Different paragraphs should be treated as a break
        similarities.append(sim)
    
    # Find potential break points (low similarity between sentences)
    break_points = [i for i, sim in enumerate(similarities) if sim < similarity_threshold]
    
    # Now chunk the text using similarity and word count constraints
    current_sentences = []
    current_word_count = 0
    
    for i, sentence in enumerate(all_sentences):
        sent_word_count = count_words(sentence)
        
        # Check if adding this sentence exceeds the word limit
        if current_word_count + sent_word_count > max_words:
            # Create a chunk with current sentences
            if current_sentences:
                chunks.append(" ".join(current_sentences))
                
                # Calculate overlap sentences
                overlap_sentences = []
                overlap_word_count = 0
                
                # Work backwards to create overlap that meets word count requirements
                for sent in reversed(current_sentences):
                    if overlap_word_count < min_overlap_words:
                        overlap_sentences.insert(0, sent)
                        overlap_word_count += count_words(sent)
                    elif overlap_word_count < max_overlap_words:
                        # Only add if similarity with next sentence is high
                        if i < len(all_sentences):
                            similarity_with_next = calculate_similarity(sent, sentence)
                            if similarity_with_next > similarity_threshold:
                                overlap_sentences.insert(0, sent)
                                overlap_word_count += count_words(sent)
                    else:
                        break
                
                # Start new chunk with overlap sentences
                current_sentences = overlap_sentences.copy()
                current_word_count = overlap_word_count
            else:
                # If we can't fit even a single sentence, we have to split it
                chunks.append(sentence)
                current_sentences = []
                current_word_count = 0
                continue
        
        # Check if we're at a break point and we have enough content for a chunk
        if i > 0 and i-1 in break_points and current_word_count > max_words * 0.5:
            # Create a chunk with current sentences
            chunks.append(" ".join(current_sentences))
            
            # Calculate overlap with emphasis on similar sentences
            overlap_sentences = []
            overlap_word_count = 0
            
            # Work backwards for overlap, focusing on similar sentences
            for j, sent in enumerate(reversed(current_sentences)):
                if overlap_word_count < min_overlap_words:
                    overlap_sentences.insert(0, sent)
                    overlap_word_count += count_words(sent)
                elif overlap_word_count < max_overlap_words:
                    if i < len(all_sentences):
                        similarity_with_next = calculate_similarity(sent, sentence)
                        if similarity_with_next > similarity_threshold:
                            overlap_sentences.insert(0, sent)
                            overlap_word_count += count_words(sent)
                else:
                    break
            
            # Start new chunk with overlap sentences
            current_sentences = overlap_sentences.copy()
            current_word_count = overlap_word_count
        
        # Add current sentence to chunk
        current_sentences.append(sentence)
        current_word_count += sent_word_count
        
        # Add paragraph boundaries where needed
        if i < len(all_sentences) - 1 and all_sent_para_mapping[i] != all_sent_para_mapping[i+1]:
            current_sentences.append("\n")
    
    # Don't forget the last chunk
    if current_sentences:
        chunks.append(" ".join(current_sentences))
    
    # Print results with similarity analysis
    print(f"Total chunks: {len(chunks)}")
    print(f"Similarity method used: {similarity_method}")
    
    for i, chunk in enumerate(chunks):
        word_count = count_words(chunk)
        print(f"Chunk {i+1} word count: {word_count}")
        print(chunk)
        print("+" * 60)
        
        # Show overlap and similarity metrics with next chunk if applicable
        if i < len(chunks) - 1:
            overlap = find_sentence_overlap(chunk, chunks[i+1])
            overlap_word_count = count_words(" ".join(overlap))
            print(f"Overlap with next chunk: {overlap_word_count} words")
            
            # Calculate average similarity between sentences in the overlap
            overlap_similarities = []
            for j in range(len(overlap) - 1):
                overlap_similarities.append(calculate_similarity(overlap[j], overlap[j+1]))
            
            if overlap_similarities:
                avg_similarity = sum(overlap_similarities) / len(overlap_similarities)
                print(f"Average {similarity_method} similarity in overlap: {avg_similarity:.3f}")
                
                # Compare with other similarity method for reference
                other_method = "jaccard" if similarity_method.lower() == "cosine" else "cosine"
                other_similarities = []
                for j in range(len(overlap) - 1):
                    if other_method == "jaccard":
                        other_similarities.append(jaccard_similarity(overlap[j], overlap[j+1]))
                    else:
                        other_similarities.append(cosine_similarity(overlap[j], overlap[j+1]))
                
                if other_similarities:
                    other_avg_similarity = sum(other_similarities) / len(other_similarities)
                    print(f"For comparison - Average {other_method} similarity: {other_avg_similarity:.3f}")
    
    return chunks

def find_sentence_overlap(chunk1, chunk2):
    """Find the overlapping sentences between two adjacent chunks."""
    sentences1 = split_into_sentences(chunk1)
    sentences2 = split_into_sentences(chunk2)
    
    overlap = []
    max_check = min(len(sentences1), len(sentences2))
    
    for i in range(max_check, 0, -1):
        if sentences1[-i:] == sentences2[:i]:
            overlap = sentences2[:i]
            break
    
    return overlap

# Example usage demonstrating both similarity methods
def test_similarity_methods():
    """Test both Jaccard and Cosine similarity methods on the same paragraphs"""
    sample_paragraphs = [
        {
            "section_id": 1,
            "paragraph_id": 1,
            "paragraph": "Machine learning models are statistical algorithms that learn patterns from data. These models can identify complex relationships without explicit programming. Deep learning is a subset of machine learning using neural networks with many layers.",
            "length": 200
        },
        {
            "section_id": 1,
            "paragraph_id": 2,
            "paragraph": "Neural networks are structured like the human brain, with interconnected nodes in multiple layers. They excel at pattern recognition tasks. Convolutional neural networks are particularly effective for image processing applications.",
            "length": 180
        },
        {
            "section_id": 1,
            "paragraph_id": 3,
            "paragraph": "Climate change refers to long-term shifts in temperatures and weather patterns. Human activities have been the main driver of climate change, primarily due to burning fossil fuels like coal, oil and gas.",
            "length": 170
        }
    ]
    
    print("=== Testing with Jaccard Similarity ===")
    jaccard_chunks = chunk_paragraphs_by_similarity(
        sample_paragraphs, 
        max_words=50, 
        min_overlap_words=10, 
        max_overlap_words=15, 
        similarity_threshold=0.15,
        similarity_method="jaccard"
    )
    
    print("\n\n=== Testing with Cosine Similarity ===")
    cosine_chunks = chunk_paragraphs_by_similarity(
        sample_paragraphs, 
        max_words=50, 
        min_overlap_words=10, 
        max_overlap_words=15, 
        similarity_threshold=0.15,
        similarity_method="cosine"
    )
    
    return jaccard_chunks, cosine_chunks


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


# exit()

# from langchain_text_splitters import CharacterTextSplitter

# text_splitter = CharacterTextSplitter(
#     separator="",
#     chunk_size=384,
#     chunk_overlap=56,
#     length_function=len,
#     is_separator_regex=False,
# )
# chunk_doc1 = text_splitter.split_text(text)
# for index, elem in enumerate(chunk_doc1):
#     print(index, elem)

# from langchain_text_splitters import RecursiveCharacterTextSplitter

# print(text)
# text_splitter2 = RecursiveCharacterTextSplitter(
#     chunk_size=50,
#     chunk_overlap=1,
#     length_function=len,
#     is_separator_regex=False,
#     separators=["\n\n", "\n", " ", ""],
# )

# chunk_doc2 = text_splitter2.split_text(text)

# for i , _ in enumerate(chunk_doc2):
#     print(f"chunk #{i} , size:{len(chunk_doc2[i])}")
#     print(chunk_doc2[i])
#     print("-----")
# print("sssssemantic lllllll")
# from sentence_transformers import SentenceTransformer  # noqa: E402
# import matplotlib.pyplot as plt
# from transformers import AutoTokenizer
# import pandas as pd


# embedding_model = "Lajavaness/bilingual-embedding-large"

# my_embedding_model = SentenceTransformer(embedding_model, trust_remote_code=True)

# tokenizer = AutoTokenizer.from_pretrained(embedding_model, trust_remote_code=True)


# def plot_chunk(chunk_doc, embedding_name):
#     tokenizer = AutoTokenizer.from_pretrained(embedding_model, trust_remote_code=True)
#     length = [len(tokenizer.encode(doc.page_content)) for doc in chunk_doc2]
#     fig = pd.Series(length).hist()
#     plt.show()


# content = text

# split_char = [".", "?", "!"]

# import re  # noqa: E402

# sentences_list = re.split(r"(?<=[.?!])\s+", content)
# print(sentences_list)
# print(len(sentences_list))

# sentences = [{"sentence": x, "index": i} for i, x in enumerate(sentences_list)]
# print("Sssssentences")
# print(sentences)
# exit()


# def combine_sentences(sentences, buffer_size=1):
#     combine_sentences = [
#         " ".join(
#             sentences[j]["sentence"]
#             for j in range(
#                 max(i - buffer_size, 0), min(i + buffer_size + 1, len(sentences))
#             )
#         )
#         for i in range(len(sentences))
#     ]

#     for i, combine_sentence in enumerate(combine_sentences):
#         sentences[i]["combined_sentence"] = combine_sentence

#     return sentences


# sentences = combine_sentences(sentences=sentences)

# print(sentences)
# print(len(sentences))
# embeddings = my_embedding_model.encode(
#     sentences=[x["combined_sentence"] for x in sentences]
# )
# for i, elem in enumerate(sentences):
#     print(elem)
#     elem["combined_sentence_embedding"] = embeddings[i]

# print("combinedddddd")
# for elem in sentences:
#     print(elem)
# print("combinedddddd___________")

# import numpy as np


# def cosine_similarity(vec1:np.ndarray, vec2:np.ndarray) -> float:
#     # Calculate the similarity between two vectors
#     dot_product = np.dot(vec1, vec2)
#     norm_vec1 = np.linalg.norm(vec1)
#     norm_vec2 = np.linalg.norm(vec2)
#     return dot_product / (norm_vec1 * norm_vec2)


# def calculate_cosine_distances(sentences):
#     distances = []
#     for i in range(len(sentences) - 1):
#         embedding_current = sentences[i]["combined_sentence_embedding"]
#         embedding_next = sentences[i + 1]["combined_sentence_embedding"]

#         # Calculate cosine similarity
#         similarity = cosine_similarity(embedding_current, embedding_next)

#         distance = 1 - similarity
#         distances.append(distance)

#         # Store into the sentences
#         sentences[i]["distance_to_next"] = distance
#     return distances, sentences


# distances, sentences = calculate_cosine_distances(sentences=sentences)

# print(sentences[-2]["distance_to_next"])

# y_upper_bound = 0.25
# plt.ylim(0, y_upper_bound)
# plt.xlim(0, len(distances))

# # percentile()
# brekpoint_percentile_threshold = 95
# brekpoint_distance_threshold = np.percentile(distances, brekpoint_percentile_threshold)

# plt.axhline(y=brekpoint_distance_threshold, color="r", linestyle="-")
# num_distances_above_threshold = len(
#     [x for x in distances if x > brekpoint_distance_threshold]
# )
# plt.text(
#     x=(len(distances) * 0.01),
#     y=y_upper_bound / 50,
#     s=f"{num_distances_above_threshold +1} chunks",
# )

# indices_above_thresh = [
#     i for i, x in enumerate(distances) if x > brekpoint_distance_threshold
# ]

# colors = ["b", "g", "r", "c", "m", "y", "k"]

# for i, breakpoint_index in enumerate(indices_above_thresh):
#     start_index = 0 if i == 0 else indices_above_thresh[i - 1]
#     end_index = (
#         breakpoint_index if i <= len(indices_above_thresh) - 1 else len(distances)
#     )

#     plt.axvspan(start_index, end_index, facecolor=colors[i % len(colors)], alpha=0.25)
#     plt.text(
#         x=np.average([start_index, end_index]),
#         y=brekpoint_distance_threshold + (y_upper_bound) / 20,
#         s=f"Chunk #{i}",
#         horizontalalignment="center",
#         rotation="vertical",
#     )

#     if indices_above_thresh:
#         last_breakpoint = indices_above_thresh[-1]
#         if last_breakpoint < len(distances):
#             plt.axvspan(
#                 last_breakpoint,
#                 len(distances),
#                 facecolor=colors[len(indices_above_thresh) % len(colors)],
#                 alpha=0.25,
#             )
#             plt.text(
#                 x=np.average([last_breakpoint, len(distances)]),
#                 y=brekpoint_distance_threshold + (y_upper_bound) / 20,
#                 s=f"Chunk #{i+1}",
#                 horizontalalignment="center",
#                 rotation="vertical",
#             )

# plt.plot(distances)
# plt.show()

# start_index = 0
# chunks = []

# for index in indices_above_thresh:
#     end_index = index
#     group = sentences[start_index : end_index + 1]
#     combined_text = " ".join([d["sentence"] for d in group])
#     chunks.append(combined_text)

#     start_index = index + 1


# if start_index < len(sentences):
#     combine_sentences = " ".join([d["sentence"] for d in sentences[start_index:]])
#     chunks.append(combined_text)

# for i, chunk in enumerate(chunks[:1]):
#     buffer = 200
#     print(f"Chunk #{i}")
#     print(chunk[:buffer].strip())
#     print("...")
#     print(chunk[-buffer:].strip())
#     print("\n")


# for i, chunk in enumerate(chunks):
#     print(i, chunk)

# print(chunks)

if __name__ == "__main__":
    main()