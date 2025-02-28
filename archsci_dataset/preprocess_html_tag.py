import re
from html import unescape

def extract_text_regex(html_text):
    """
    Extract text content from HTML using regular expressions.
    
    This approach is faster but less reliable for complex HTML structures.
    
    Args:
        html_text (str): HTML content as string
        
    Returns:
        str: Plain text with HTML removed but content preserved
    """
    # First, handle common HTML entities
    text = unescape(html_text)
    
    # Remove script and style elements (including their content)
    text = re.sub(r'<script.*?>.*?</script>', ' ', text, flags=re.DOTALL)
    text = re.sub(r'<style.*?>.*?</style>', ' ', text, flags=re.DOTALL)
    
    # Replace line breaks and paragraph tags with newlines
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</?p>', '\n', text, flags=re.IGNORECASE)
    
    # Replace consecutive spaces (including non-breaking spaces) with a single space
    text = re.sub(r'(&nbsp;|\s)+', ' ', text)
    
    # Remove all remaining HTML tags
    text = re.sub(r'<[^>]*>', '', text)
    
    # Clean up whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Replace multiple newlines with just two
    text = re.sub(r'^\s+', '', text)  # Remove leading whitespace
    text = re.sub(r'\s+$', '', text)  # Remove trailing whitespace
    
    return text