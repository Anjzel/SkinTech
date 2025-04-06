import sys
import pandas as pd
import re

try:
    import pyarrow as pa
except ImportError:
    print("Warning: pyarrow not installed. Using default CSV engine.")
    print("To install pyarrow, run: pip install pyarrow")

def extract_ingredients(description):
    # List of keywords to identify ingredients
    ingredient_indicators = [
        r'contains?|enriched?|with|based|formula contains|ingredients?|active'
    ]
    
    # Common ingredients to look for
    ingredients = set()
    
    # Find sentences containing ingredient indicators
    for indicator in ingredient_indicators:
        matches = re.finditer(indicator, description.lower())
        for match in matches:
            # Get the text after the indicator
            start = match.end()
            sentence_end = description[start:].find('.')
            if sentence_end == -1:
                continue
            
            ingredient_text = description[start:start+sentence_end]
            
            # Extract individual ingredients
            # Look for words following common patterns
            found = re.findall(r'(?:[\w-]+\s*)+(?:acid|extract|oil|water|vitamin|zinc|copper|glycols?|butter)', ingredient_text, re.IGNORECASE)
            ingredients.update(found)
    
    return ', '.join(ingredients) if ingredients else ''

try:
    # Read the CSV file with default engine
    df = pd.read_csv('c:/Users/richard/Downloads/cleaned_products.csv')
    
    # Extract ingredients from description
    df['ingredients'] = df['description'].apply(extract_ingredients)
    
    # Save updated CSV with default engine
    df.to_csv('c:/Users/richard/Downloads/cleaned_products.csv', index=False)
    print("Successfully extracted ingredients!")

except Exception as e:
    print(f"Error: {str(e)}")
    sys.exit(1)
