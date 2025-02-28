import re

def preprocess_footnote_v1(text: str) -> str:
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


def preprocess_footnote(text:str) -> str:
    # Regex pattern with DOTALL flag to include newlines 
    footnote_pattern = re.compile(r'^\d+\s.*\.', re.DOTALL)
    match = footnote_pattern.search(text)
    if match:
        return ""
    else:
        return text

def preprocess_footnote_intext_replace(text:str, matched_list:list) -> str:
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
        '''.format('|'.join(matched_list)),
        flags=re.VERBOSE
    ) 
    
    # Remove the footnote numbers from the text
    clean_text = footnote_pattern.sub('', text)
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

def preprocess_footnotes_file(filename: str) -> None:
    with open(filename) as fp: 
        content = fp.readlines()
    text = "" 
    footnotes_identified = [] 
    last_footnotes = '' 
    for line in content: 
        tmp_str = preprocess_footnote(line) 
        clean_text = '' 
        if tmp_str != "": 
            matches_footnote = preprocess_footnote_intext(line) 
            if len(footnotes_identified) > 0: 
                last_footnotes = footnotes_identified[-1]
            if len(matches_footnote) > 0:
                if matches_footnote[0] > last_footnotes: 
                    clean_text = preprocess_footnote_intext_replace(line, matches_footnote)
                    footnotes_identified.extend(matches_footnote)
            
            if len(clean_text) > 0: 
                text += clean_text 
            else: 
                text += line
    print(text)