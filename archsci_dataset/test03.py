from collections import deque
import numpy as np
from sentence_transformers import SentenceTransformer
import re
from archsci_dataset.preprocess_footnotes import (
    preprocess_footnote_v1,
    preprocess_footnote_intext,
    preprocess_footnote_intext_replace,
    preprocess_footnote,
)
from archsci_dataset.similarity import *
from archsci_dataset.generate_chunks import (
    split_into_section_paragraphs,
    chunk_paragraphs_with_overlap,
    count_words,
)
from collections import Counter
from loguru import logger
from sentence_transformers import SentenceTransformer, util
from archsci_dataset.preprocess_html_tag import extract_text_regex
#from langchain_community.embeddings import HuggingFaceEmbeddings


def cosine_similarity_embed(sentences_embed) -> np.ndarray:
    """
    Calculates the cosine distances between consecutive text chunk embeddings.

    Returns:
        np.ndarray: An array of cosine distances between consecutive embeddings.
    """
    len_embeddings = len(sentences_embed)
    cdists = np.empty(len_embeddings - 1)

    for i in range(1, len_embeddings):
        cdists[i - 1] = util.cos_sim(sentences_embed[i], sentences_embed[i - 1])

    return cdists

def semantic_chunks(sentences, break_points: list[int]) -> list[str]:
    """
    Constructs semantic groups from text splits using specified breakpoints.

    Args:
        breakpoints (List[int]): A list of indices representing breakpoints.

    Returns:
        List[str]: A list of concatenated text strings for each semantic group.
    """
    start_index = 0
    grouped_texts = []
    # add end criteria
    breakpoints = np.append(break_points, [-1])
    for break_point in breakpoints:
        # we're at the end of the text
        if break_point == -1:
            grouped_texts.append(
                " ".join([x for x in sentences[start_index:]])
            )

        else:
            grouped_texts.append(
                " ".join([x for x in sentences[start_index : break_point + 1]])
            )

        start_index = break_point + 1

    return grouped_texts

def chunk_breakpoints(similarities: list[np.ndarray], breakpoint_percentile_threshold:int=95):
    """
    Identify indices in the similarity list where the similarity values exceed a certain percentile threshold.

    Args:
        similarities (list[np.ndarray]): A list of similarity scores between consecutive text chunks.
        breakpoint_percentile_threshold (int): The percentile threshold to determine breakpoints. Default is 95.

    Returns:
        np.ndarray: An array of indices where the similarity values exceed the calculated threshold.
    """
    breakpoint_distance_threshold = np.percentile(similarities, 80)
    threshold_indices = np.argwhere(similarities>= breakpoint_distance_threshold).ravel()
    return threshold_indices
    

def build_chunks_stack(distances, 
                       all_sentences,
                       length_threshold: int = 384, 
                       cosine_distance_percentile_threshold: int = 95,
) -> np.ndarray:
    
    S = [(0, len(distances))]
    all_breakpoints = set()
    while S:
        id_start, id_end = S.pop()
        distance = distances[id_start:id_end]
        updated_breakpoints = chunk_breakpoints(
            distance,
            breakpoint_percentile_threshold=cosine_distance_percentile_threshold,
        )
        if updated_breakpoints.size == 0:
            continue  
        updated_breakpoints += id_start
        updated_breakpoints = np.concatenate(
            (np.array([id_start - 1]), updated_breakpoints, np.array([id_end]))
        )
        for index in updated_breakpoints:
            text_group = all_sentences[id_start : index + 1]
            total_text = sum(len(text) for text in text_group)
            if (len(text_group) > 2) and (
                total_text >= length_threshold
            ):
                S.append((id_start, index))
            id_start = index + 1
        all_breakpoints.update(updated_breakpoints)

    return np.array(sorted(all_breakpoints))[1:-1]

def chunk_by_transformer_similarity(paras, max_words=384, min_overlap_words=56, max_overlap_words=96, 
                                 similarity_threshold=0.8, model_name='all-MiniLM-L6-v2'):
    """
    Split paragraphs into chunks based on sentence transformer similarity and word count constraints.
    Uses pretrained language models to understand semantic relationships between sentences.
    
    Args:
        paras (list): List of paragraph dictionaries with keys 'section_id', 'paragraph_id', 
                      'paragraph', and 'length'
        max_words (int): Maximum number of words in each chunk
        min_overlap_words (int): Minimum number of words to overlap between chunks
        max_overlap_words (int): Maximum number of words to overlap between chunks
        similarity_threshold (float): Threshold for determining sentence similarity (0-1)
        model_name (str): Name of the sentence-transformer model to use
        
    Returns:
        list: A list of text chunks with sentences grouped by semantic similarity
    """
    chunks = []

    # Load the sentence transformer model
    logger.info(f"Loading sentence transformer model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Extract all sentences from all paragraphs
    all_sentences = []
    para_boundaries = []  # Track where paragraph boundaries occur
    
    # Collect all sentences
    for para_idx, para in enumerate(paras):
        logger.info(f"{para['section_id']}, {para['paragraph_id']}, Word count: {para['length']}")
        
        # Split paragraph into sentences
        paragraph_sentences = split_into_sentences(para["paragraph"])
        
        # Mark paragraph boundary
        if all_sentences:
            para_boundaries.append(len(all_sentences))
            
        # Add sentences from this paragraph
        all_sentences.extend(paragraph_sentences)
    
    # Get embeddings for all sentences at once (more efficient)
    logger.info("Encoding sentences with transformer model...")
    sentence_embeddings = model.encode(all_sentences, show_progress_bar=True)

    # Calculate cosine similarity between adjacent sentences
    similarities = cosine_similarity_embed(sentence_embeddings)
    
    # Identify potential break points (places with low sentence similarity)
    break_points = []
    threshold_indices = chunk_breakpoints(similarities, 80)
    print(threshold_indices)
    chunks = semantic_chunks(all_sentences, threshold_indices)
    print(chunks)
    for chunk in chunks:
        print(len(chunk), chunk)
        print("=" * 60)
    new_threshold = build_chunks_stack(similarities, all_sentences,cosine_distance_percentile_threshold=80)
    print(new_threshold)
    chunks1 = semantic_chunks(all_sentences, new_threshold)
    for chunk in chunks1:
        print(len(chunk), chunk)
        print("=" * 60)
    exit()

    for i, sim in enumerate(similarities):
        if sim < similarity_threshold:
            break_points.append(i)

    print(break_points) 
    # Also consider paragraph boundaries as natural break points
    for boundary in para_boundaries:
        if boundary - 1 not in break_points and boundary - 1 >= 0:  # -1 because similarities array is offset
            break_points.append(boundary - 1)
    print(break_points) 
    
    # Sort break points
    break_points.sort()
    logger.info(f"Identified {len(break_points)} potential break points based on similarity threshold {similarity_threshold}")
    
    print(break_points) 
    exit()
    # Now create chunks
    current_sentences = []
    current_word_count = 0
    last_break_point = -1
    
    for i, sentence in enumerate(all_sentences):
        sent_word_count = count_words(sentence)
        
        # Check if adding this sentence would exceed the word limit
        if current_word_count + sent_word_count > max_words:
            # This sentence would make the chunk too large
            if current_sentences:
                # Save current chunk
                chunks.append(" ".join(current_sentences))
                
                # Create overlap based on similarity and word count
                overlap_sentences = []
                overlap_word_count = 0
                
                # First, include sentences to meet minimum overlap
                for sent in reversed(current_sentences):
                    if overlap_word_count < min_overlap_words:
                        overlap_sentences.insert(0, sent)
                        overlap_word_count += count_words(sent)
                    else:
                        break
                
                # Then, add sentences with high similarity to the next one
                if i < len(all_sentences):
                    # Calculate similarity between last overlap sentence and next sentence
                    sent_idx = all_sentences.index(overlap_sentences[-1])
                    similarity_to_next = cosine_similarity(
                        sentence_embeddings[sent_idx], 
                        sentence_embeddings[i]
                    )
                    
                    # If similarity with next sentence is high, add more from current chunk
                    if similarity_to_next > similarity_threshold and overlap_word_count < max_overlap_words:
                        possible_additions = [
                            sent for sent in current_sentences 
                            if sent not in overlap_sentences and 
                            overlap_word_count + count_words(sent) <= max_overlap_words
                        ]
                        
                        # Calculate similarity of each possible addition to the next sentence
                        similarities_to_next = []
                        for sent in possible_additions:
                            sent_idx = all_sentences.index(sent)
                            sim = cosine_similarity(sentence_embeddings[sent_idx], sentence_embeddings[i])
                            similarities_to_next.append((sent, sim))
                        
                        # Sort by similarity (highest first)
                        sorted_additions = sorted(similarities_to_next, key=lambda x: x[1], reverse=True)
                        
                        # Add highest similarity sentences
                        for sent, sim in sorted_additions:
                            if sim > similarity_threshold and overlap_word_count + count_words(sent) <= max_overlap_words:
                                # Find position in original sentences to maintain order
                                insert_pos = 0
                                for j, existing in enumerate(overlap_sentences):
                                    if current_sentences.index(existing) > current_sentences.index(sent):
                                        insert_pos = j
                                        break
                                    insert_pos = j + 1
                                
                                overlap_sentences.insert(insert_pos, sent)
                                overlap_word_count += count_words(sent)
                
                # Start a new chunk with the overlap sentences
                current_sentences = overlap_sentences.copy()
                current_word_count = overlap_word_count
                
                # Try to add the current sentence if it fits
                if current_word_count + sent_word_count <= max_words:
                    current_sentences.append(sentence)
                    current_word_count += sent_word_count
                else:
                    # This sentence doesn't fit - start another chunk
                    chunks.append(" ".join(current_sentences))
                    current_sentences = [sentence]
                    current_word_count = sent_word_count
            else:
                # Edge case: a single sentence is longer than max_words
                chunks.append(sentence)
                current_sentences = []
                current_word_count = 0
            
            # Reset last break point
            last_break_point = i - 1
            continue
        
        # Check if we're at a good break point and have enough content for a chunk
        if i > 0 and i - 1 in break_points and i - 1 > last_break_point and current_word_count > max_words * 0.5:
            # This is a good place to break because similarity is low
            chunks.append(" ".join(current_sentences))
            
            # Calculate overlap with similar sentences
            overlap_sentences = []
            overlap_word_count = 0
            
            # First, include sentences to meet minimum overlap
            for sent in reversed(current_sentences):
                if overlap_word_count < min_overlap_words:
                    overlap_sentences.insert(0, sent)
                    overlap_word_count += count_words(sent)
                else:
                    break
            
            # Then, add sentences with high similarity to the current one
            if overlap_sentences:
                sent_idx = all_sentences.index(overlap_sentences[-1])
                current_idx = i
                similarity_to_current = cosine_similarity(
                    sentence_embeddings[sent_idx], 
                    sentence_embeddings[current_idx]
                )
                
                if similarity_to_current > similarity_threshold and overlap_word_count < max_overlap_words:
                    possible_additions = [
                        sent for sent in current_sentences 
                        if sent not in overlap_sentences and 
                        overlap_word_count + count_words(sent) <= max_overlap_words
                    ]
                    
                    # Calculate similarity of each possible addition to the current sentence
                    similarities_to_current = []
                    for sent in possible_additions:
                        sent_idx = all_sentences.index(sent)
                        sim = cosine_similarity(sentence_embeddings[sent_idx], sentence_embeddings[current_idx])
                        similarities_to_current.append((sent, sim))
                    
                    # Sort by similarity
                    sorted_additions = sorted(similarities_to_current, key=lambda x: x[1], reverse=True)
                    
                    # Add highest similarity sentences
                    for sent, sim in sorted_additions:
                        if sim > similarity_threshold and overlap_word_count + count_words(sent) <= max_overlap_words:
                            insert_pos = 0
                            for j, existing in enumerate(overlap_sentences):
                                if current_sentences.index(existing) > current_sentences.index(sent):
                                    insert_pos = j
                                    break
                                insert_pos = j + 1
                            
                            overlap_sentences.insert(insert_pos, sent)
                            overlap_word_count += count_words(sent)
            
            # Start new chunk with overlap sentences
            current_sentences = overlap_sentences.copy()
            current_word_count = overlap_word_count
            
            # Add current sentence
            current_sentences.append(sentence)
            current_word_count += sent_word_count
            
            # Mark this as processed
            last_break_point = i - 1
            continue
        
        # Add current sentence to the chunk
        current_sentences.append(sentence)
        current_word_count += sent_word_count
        
        # Add paragraph markers where needed (for display purposes)
        if i < len(all_sentences) - 1 and i in para_boundaries:
            current_sentences.append("\n")
    
    # Add the final chunk
    if current_sentences:
        chunks.append(" ".join(current_sentences))
    
    # Print chunks with similarity information
    print(f"\nTotal chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        word_count = count_words(chunk)
        print(f"\nChunk {i+1} word count: {word_count}")
        print(chunk)
        print("=" * 60)
        
        # Show overlap with next chunk
        if i < len(chunks) - 1:
            overlap = find_sentence_overlap(chunk, chunks[i+1])
            if overlap:
                overlap_text = " ".join(overlap)
                overlap_word_count = count_words(overlap_text)
                print(f"Overlap with next chunk: {overlap_word_count} words ({len(overlap)} sentences)")
                
                # Visualize similarity between sentences in overlap
                if len(overlap) > 1:
                    print("Similarities within overlap:")
                    for j in range(len(overlap) - 1):
                        idx1 = all_sentences.index(overlap[j])
                        idx2 = all_sentences.index(overlap[j+1])
                        sim = cosine_similarity(sentence_embeddings[idx1], sentence_embeddings[idx2])
                        print(f"  {j}->{j+1}: {sim:.4f}")
    
    return chunks

def cosine_similarity(vector1, vector2):
    """Calculate cosine similarity between two vectors."""
    # Ensure the vectors are numpy arrays
    v1 = np.array(vector1)
    v2 = np.array(vector2)
    
    # Calculate cosine similarity
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0
        
    return dot_product / (norm1 * norm2)

def find_sentence_overlap(chunk1, chunk2):
    """Find the overlapping sentences between two adjacent chunks."""
    sentences1 = split_into_sentences(chunk1)
    sentences2 = split_into_sentences(chunk2)
    
    # Filter out any newline markers
    sentences1 = [s for s in sentences1 if s != "\n"]
    sentences2 = [s for s in sentences2 if s != "\n"]
    
    # Look for exact sentence matches
    overlap = []
    
    # Try to find the longest sequence of matching sentences
    max_possible = min(len(sentences1), len(sentences2))
    
    for i in range(max_possible, 0, -1):
        if sentences1[-i:] == sentences2[:i]:
            overlap = sentences2[:i]
            break
    
    return overlap

def split_into_sentences(text):
    """Split text into sentences while handling common abbreviations."""
    # Handle common abbreviations to avoid false sentence breaks
    text = re.sub(r'Mr\.', 'Mr_DOT_', text)
    text = re.sub(r'Mrs\.', 'Mrs_DOT_', text)
    text = re.sub(r'Dr\.', 'Dr_DOT_', text)
    text = re.sub(r'Ms\.', 'Ms_DOT_', text)
    text = re.sub(r'Prof\.', 'Prof_DOT_', text)
    text = re.sub(r'Inc\.', 'Inc_DOT_', text)
    text = re.sub(r'Ltd\.', 'Ltd_DOT_', text)
    text = re.sub(r'i\.e\.', 'i_e_DOT_', text)
    text = re.sub(r'e\.g\.', 'e_g_DOT_', text)
    
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z]|$)', text)
    
    # Restore abbreviations
    sentences = [re.sub(r'Mr_DOT_', 'Mr.', s) for s in sentences]
    sentences = [re.sub(r'Mrs_DOT_', 'Mrs.', s) for s in sentences]
    sentences = [re.sub(r'Dr_DOT_', 'Dr.', s) for s in sentences]
    sentences = [re.sub(r'Ms_DOT_', 'Ms.', s) for s in sentences]
    sentences = [re.sub(r'Prof_DOT_', 'Prof.', s) for s in sentences]
    sentences = [re.sub(r'Inc_DOT_', 'Inc.', s) for s in sentences]
    sentences = [re.sub(r'Ltd_DOT_', 'Ltd.', s) for s in sentences]
    sentences = [re.sub(r'i_e_DOT_', 'i.e.', s) for s in sentences]
    sentences = [re.sub(r'e_g_DOT_', 'e.g.', s) for s in sentences]
    
    return sentences

# Example usage with installation instructions
def test_transformer_chunking():
    """Test the transformer-based similarity chunking with sample paragraphs"""
    # Installation instructions
    print("Before running this code, you need to install sentence-transformers:")
    print("pip install sentence-transformers")
    print("\n")
    
    # Sample paragraphs with different topics
    sample_paragraphs = [
        {
            "section_id": 1,
            "paragraph_id": 1,
            "paragraph": "Machine learning algorithms build models from example data. These models learn patterns in the training data. The models can then make predictions on new data. This approach differs from traditional rule-based programming.",
            "length": 200
        },
        {
            "section_id": 1,
            "paragraph_id": 2,
            "paragraph": "Deep learning is a subset of machine learning. It uses neural networks with many layers. These networks can find complex patterns in data. They have revolutionized computer vision and natural language processing.",
            "length": 180
        },
        {
            "section_id": 1,
            "paragraph_id": 3,
            "paragraph": "The climate crisis demands immediate action. Rising temperatures threaten ecosystems worldwide. Polar ice caps are melting at an alarming rate. Sea levels are expected to rise significantly in the coming decades.",
            "length": 170
        }
    ]
    
    # For better results with sentence transformers, you may want to use a more powerful model:
    # all-MiniLM-L6-v2 (Default) - Fast, good quality
    # all-mpnet-base-v2 - Slower, higher quality
    # paraphrase-multilingual-MiniLM-L12-v2 - Good for non-English text
    
    print("=== Testing with Transformer-Based Sentence Similarity ===")
    chunks = chunk_by_transformer_similarity(
        sample_paragraphs, 
        max_words=50, 
        min_overlap_words=10, 
        max_overlap_words=15, 
        similarity_threshold=0.7,  # Higher threshold for transformer models (they produce higher similarities)
        model_name='all-MiniLM-L6-v2'  # Lightweight default model
    )
    
    return chunks

def main():
    with open("./data/journals/00003.md") as fp:
        content = fp.readlines()

    text = ""
    footnotes_identified = []
    last_footnotes = ''
    for line in content:
        tmp_str = preprocess_footnote_v1(line)
        clean_text = ''
        if tmp_str != "": 
            matches_footnote = preprocess_footnote_intext(line)
            logger.info("=" * 60)
            logger.info(matches_footnote)
            logger.info("===== 1 =========")
            logger.info(footnotes_identified)
            if len(footnotes_identified) > 0: 
                last_footnotes = footnotes_identified[-1]
            logger.info("===== 2 =========")
            logger.info(last_footnotes)
            if len(matches_footnote) > 0:
                if matches_footnote[0] > last_footnotes: 
                    clean_text = preprocess_footnote_intext_replace(line, matches_footnote)
                    footnotes_identified.extend(matches_footnote)
                
            if len(clean_text) > 0: 
                text += extract_text_regex(clean_text)
            else:
                text += extract_text_regex(line)

    print("----------- 1. preprocess footnote ============")
    print(text)
    # 1. preprocess_footnote


    paras = split_into_section_paragraphs(text) 
    #chunks = chunk_paragraphs(paras) 
    
    chunks_overlap = chunk_paragraphs_with_overlap(paras)
    print(chunks_overlap)
    print("=== Testing with Transformer-Based Sentence Similarity ===")
    chunks = chunk_by_transformer_similarity(
        paras, 
        max_words=256, 
        min_overlap_words=16, 
        max_overlap_words=32, 
        similarity_threshold=0.8,  # Higher threshold for transformer models (they produce higher similarities)
        model_name='all-MiniLM-L6-v2'  # Lightweight default model
    )