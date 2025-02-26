import re

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