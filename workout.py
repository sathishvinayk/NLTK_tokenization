import requests
from bs4 import BeautifulSoup
import re
import os
import nltk
from nltk.tokenize import RegexpTokenizer, sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import pandas as pd

def setup_nltk():
    """Ensure all required NLTK resources are available"""
    try:
        # Download stopwords if not found
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            print("Downloading NLTK stopwords...")
            nltk.download('stopwords')
        
        # Download punkt for sentence tokenization
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading NLTK punkt...")
            nltk.download('punkt')
        
        # Download punkt_tab for English sentence tokenization
        try:
            nltk.data.find('tokenizers/punkt/english.pickle')
        except LookupError:
            print("Downloading NLTK punkt English data...")
            nltk.download('punkt_tab')
    except Exception as e:
        print(f"Error during NLTK setup: {e}")
        raise  # Re-raise the exception to stop execution if setup fails

# Initialize NLTK
setup_nltk()

def read_urls_from_excel(file_path, url_col='URL', url_id_col='URL_ID'):
    try:
        # Read the Excel file
        df = pd.read_excel(file_path)
        
        # Extract URL and URL_ID columns
        urls = df[url_col].tolist()
        url_ids = df[url_id_col].tolist()
        
        # Return as list of tuples
        return list(zip(urls, url_ids))
    
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return []

def clean_text(text, stop_words):
    """Clean text using regex tokenizer (no punkt dependency)"""
    if not text:
        return []
    
    try:
        # Create a regex tokenizer that keeps only words
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(text.lower())
        return [word for word in tokens if word.isalpha() and word not in stop_words]
    except Exception as e:
        print(f"Error during text cleaning: {e}")
        return []

def load_stop_words(stop_words_dir):
    stop_words = set()
    for filename in os.listdir(stop_words_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(stop_words_dir, filename), 'r', encoding='utf-8', errors='ignore') as file:
                stop_words.update(set(file.read().split()))
    return stop_words

def load_master_dictionary(positive_file, negative_file, stop_words):
    with open(positive_file, 'r', encoding='utf-8', errors='ignore') as file:
        positive_words = set(word.strip() for word in file.read().split() if word.strip() not in stop_words)

    with open(negative_file, 'r', encoding='utf-8', errors='ignore') as file:
        negative_words = set(word.strip() for word in file.read().split() if word.strip() not in stop_words)

    return positive_words, negative_words

def count_syllables(word):
    """Count syllables in a word"""
    word = word.lower()
    vowels = "aeiouy"
    count = 0
    prev_char_was_vowel = False
    
    # Handle some exceptions
    if word.endswith(('es', 'ed')) and len(word) > 2:
        word = word[:-2]
    
    for char in word:
        if char in vowels and not prev_char_was_vowel:
            count += 1
            prev_char_was_vowel = True
        else:
            prev_char_was_vowel = False
    
    return max(1, count)  # Every word has at least one syllable

def count_complex_words(tokens):
    """Count words with more than two syllables"""
    return sum(1 for word in tokens if count_syllables(word) > 2)

def count_personal_pronouns(text):
    """Count personal pronouns using regex"""
    # Pattern to match personal pronouns (case insensitive)
    pattern = r'\b(I|we|my|ours|us)\b'
    # Exclude 'US' (country) by checking for word boundaries
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    # Filter out 'US' country references
    filtered = [m for m in matches if m.lower() != 'us' or 
               (m == 'us' and not re.search(r'\bUS\b', text))]
    return len(filtered)

def calculate_readability_scores(text, tokens):
    """Calculate all readability metrics"""
    try:
        sentences = sent_tokenize(text)
    except:
        # Fallback to simple sentence splitting if punkt fails
        sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    
    num_sentences = len(sentences)
    num_words = len(tokens)
    num_complex_words = count_complex_words(tokens)
    
    #FORMULAE
    # Average Sentence Length
    avg_sentence_length = num_words / num_sentences if num_sentences > 0 else 0
    
    # Percentage of Complex Words
    percentage_complex = (num_complex_words / num_words) * 100 if num_words > 0 else 0
    
    # Fog Index
    fog_index = 0.4 * (avg_sentence_length + percentage_complex)
    
    # Average Number of Words Per Sentence
    avg_words_per_sentence = avg_sentence_length  # Same as above
    
    # Word Count (already cleaned, without stop words)
    word_count = num_words
    
    # Syllable Count Per Word
    syllable_counts = [count_syllables(word) for word in tokens]
    avg_syllables_per_word = sum(syllable_counts) / num_words if num_words > 0 else 0
    
    # Personal Pronouns
    personal_pronouns = count_personal_pronouns(text)
    
    # Average Word Length
    avg_word_length = sum(len(word) for word in tokens) / num_words if num_words > 0 else 0
    
    return {
        'avg_sentence_length': avg_sentence_length,
        'percentage_complex_words': percentage_complex,
        'fog_index': fog_index,
        'avg_words_per_sentence': avg_words_per_sentence,
        'complex_word_count': num_complex_words,
        'word_count': word_count,
        'avg_syllables_per_word': avg_syllables_per_word,
        'personal_pronouns': personal_pronouns,
        'avg_word_length': avg_word_length
    }

def calculate_sentiment_scores(tokens, positive_words, negative_words):
    positive_score = sum(1 for word in tokens if word in positive_words)
    negative_score = sum(1 for word in tokens if word in negative_words)

    polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)
    subjectivity_score = (positive_score + negative_score) / (len(tokens) + 0.000001)

    return {
        'positive_score': positive_score,
        'negative_score': negative_score,
        'polarity_score': polarity_score,
        'subjectivity_score': subjectivity_score,
    }

def extract_article(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors

        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.find('h1', class_='entry-title').get_text(strip=True)
        
        content_div = soup.find('div', class_='td-post-content')

        for element in content_div.find_all(['div', 'script', 'style', 'iframe', 'ins']):
            element.decompose()
        
        article_text = content_div.get_text('\n', strip=True)

        article_text = re.sub(r'\n\s*\n', '\n\n', article_text)
        
        return title, article_text

    except Exception as e:
        print(f"Error extracting article: {e}")
        return None, None

def analyze_article(url, url_id, stop_words, positive_words, negative_words):
    title, text = extract_article(url)
    if title and text:
        cleaned_tokens = clean_text(text, stop_words)
        
        # Calculate sentiment scores
        sentiment_scores = calculate_sentiment_scores(cleaned_tokens, positive_words, negative_words) # Until u understood
        
        # ------ XXXXX ------# 

        # Calculate readability scores
        readability_scores = calculate_readability_scores(text, cleaned_tokens) # FROM point 2
        
        # Combine all results
        results = {
            'url_id': url_id,
            'url': url,
            'title': title,
            **sentiment_scores,
            **readability_scores
        }
        
        return results
    else:
        print(f"Failed to extract article from {url}")
        return None

if __name__ == '__main__':
    STOP_WORDS_DIR = 'StopWords'
    POSITIVE_DICT = 'MasterDictionary/positive-words.txt'
    NEGATIVE_DICT = 'MasterDictionary/negative-words.txt'

    url = "https://insights.blackcoffer.com/ai-and-ml-based-youtube-analytics-and-content-creation-tool-for-optimizing-subscriber-engagement-and-content-strategy/"
    url_id = "Netclan20241017"

    stop_words = load_stop_words(STOP_WORDS_DIR)
    positive_words, negative_words = load_master_dictionary(POSITIVE_DICT, NEGATIVE_DICT, stop_words)

    results = analyze_article(url, url_id, stop_words, positive_words, negative_words)
    if results:
        print("Analysis Results:")
        for key, value in results.items():
            print(f"{key}: {value}")