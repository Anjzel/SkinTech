import re
import pandas as pd

def clean_ingredient_word(word):
    # Remove special characters and extra spaces
    word = re.sub(r'[^\w\s-]', '', word.lower()).strip()
    
    # Remove common stop words and descriptors
    stop_words = ['and', 'with', 'contains', 'extract', 'derived', 'from', 'based', 'concentrate']
    if word in stop_words:
        return ''
        
    return word

def parse_ingredients(text):
    if pd.isna(text):
        return []
        
    # Split by common separators
    words = re.split(r'[,;()]', text)
    
    # Clean each word
    cleaned_words = []
    for word in words:
        word = clean_ingredient_word(word)
        if word:
            cleaned_words.append(word)
            
    return list(set(cleaned_words)) # Remove duplicates

def convert_to_string(ingredients):
    return ', '.join(ingredients)
