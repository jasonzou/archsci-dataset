import re
from collections import deque
import numpy as np
from sentence_transformers import SentenceTransformer
from archsci_dataset.generate_chunks import split_into_section_paragraphs


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

def semantic_chunk():
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

  paras = split_into_section_paragraphs(text)
  #chunks = chunk_paragraphs(paras)
  chunks_overlap = chunk_paragraphs_with_overlap(paras)

def chunk_by_transformer_similarity(paras, max_words=384, min_overlap_words=56, max_overlap_words=96, 
                                 similarity_threshold=0.7, model_name='all-MiniLM-L6-v2'):
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
    
    def count_words(text):
        """Count the number of words in a text string."""
        return len(text.split())
    
    # Load the sentence transformer model
    print(f"Loading sentence transformer model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Extract all sentences from all paragraphs
    all_sentences = []
    para_boundaries = []  # Track where paragraph boundaries occur
    
    # Collect all sentences
    for para_idx, para in enumerate(paras):
        print(f"{para['section_id']}, {para['paragraph_id']}, Word count: {count_words(para['paragraph'])}")
        
        # Split paragraph into sentences
        paragraph_sentences = split_into_sentences(para["paragraph"])
        
        # Mark paragraph boundary
        if all_sentences:
            para_boundaries.append(len(all_sentences))
            
        # Add sentences from this paragraph
        all_sentences.extend(paragraph_sentences)
    
    # Get embeddings for all sentences at once (more efficient)
    print("Encoding sentences with transformer model...")
    sentence_embeddings = model.encode(all_sentences, show_progress_bar=True)
    
    # Calculate cosine similarity between adjacent sentences
    similarities = []
    for i in range(len(all_sentences) - 1):
        # Don't calculate similarity across paragraph boundaries
        if i + 1 in para_boundaries:
            sim = 0  # Force low similarity at paragraph boundaries
        else:
            # Cosine similarity between sentence embeddings
            embedding1 = sentence_embeddings[i]
            embedding2 = sentence_embeddings[i+1]
            sim = cosine_similarity(embedding1, embedding2)
        similarities.append(sim)
        
    # Print some similarity statistics
    if similarities:
        print(f"Similarity statistics: min={min(similarities):.4f}, max={max(similarities):.4f}, " +
              f"avg={sum(similarities)/len(similarities):.4f}")
    
    # Identify potential break points (places with low sentence similarity)
    break_points = []
    for i, sim in enumerate(similarities):
        if sim < similarity_threshold:
            break_points.append(i)
    
    # Also consider paragraph boundaries as natural break points
    for boundary in para_boundaries:
        if boundary - 1 not in break_points and boundary - 1 >= 0:  # -1 because similarities array is offset
            break_points.append(boundary - 1)
    
    # Sort break points
    break_points.sort()
    print(f"Identified {len(break_points)} potential break points based on similarity threshold {similarity_threshold}")
    
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