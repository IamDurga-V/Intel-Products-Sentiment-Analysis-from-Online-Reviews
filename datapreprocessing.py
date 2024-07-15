import re
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin

# Define custom stopwords including new ones
stop_words = set(stopwords.words('english'))
new_stopwords = ["mario", "blah", "saturday", "monday", "sunday", "morning", "evening", "friday", "would", "shall", "could", "might"]
stop_words.update(new_stopwords)
stop_words.discard("not")  # Remove "not" from stopwords

# Define a function to remove special characters
def remove_special_characters(text):
    return re.sub(r'\W+', ' ', text)

# Define a function to remove URLs
def remove_urls(text):
    return re.sub(r'http\S+', '', text)

# Define a function for contraction expansion
def expand_contractions(text):
    # Add more contraction expansions if necessary
    contraction_patterns = {
        r"won\'t": "will not",
        r"can\'t": "can not",
        r"n\'t": " not",
        # Add more as needed
    }
    patterns = re.compile('|'.join(contraction_patterns.keys()))
    expanded_text = patterns.sub(lambda match: contraction_patterns[match.group(0)], text)
    return expanded_text

# Define a function for data cleaning
def data_cleaning(text):
    text = expand_contractions(text)
    text = remove_special_characters(text)
    text = remove_urls(text)
    
    # Tokenize text and remove stopwords
    tokens = word_tokenize(text.lower())
    cleaned_tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    
    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    cleaned_text = ' '.join([lemmatizer.lemmatize(token) for token in cleaned_tokens])
    
    return cleaned_text

# Custom transformer for data cleaning
class DataCleaning(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X.apply(data_cleaning)

# Example usage:
# clean_transformer = DataCleaning()
# cleaned_data = clean_transformer.transform(your_data)
