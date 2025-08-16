import requests
import time
from bs4 import BeautifulSoup
import re
import os
import nltk
from nltk.tokenize import RegexpTokenizer, sent_tokenize
from nltk.corpus import stopwords
import pandas as pd
from functools import lru_cache
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

CONFIG = {
    'STOPWORDS_DIRECTORY': 'StopWords',
    'POSITIVE_DICT': 'MasterDictionary/positive-words.txt',
    'NEGATIVE_DICT': 'MasterDictionary/negative-words.txt',
    'INPUT_FILE': 'Input.xlsx',
    'OUTPUT_FILE': 'Output.xlsx',
    'REQUEST_DELAY': 1.5,
    'MAX_RETRIES': 3
}

def format_metric_name(name):
    """Format metric names for output"""
    return name.replace('_', ' ').upper()

@retry(stop=stop_after_attempt(CONFIG['MAX_RETRIES']), 
       wait=wait_exponential(multiplier=1, min=4, max=10))
def setup_nltk():
    """Ensure all required NLTK resources are available"""
    try:
        # Download stopwords if not found
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            print("Downloading NLTK stopwords...")
            nltk.download('stopwords')
        
        # PUNKT_TAB is required, Sometimes its not working (adding this just in case)
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
        raise

def clean_text(text, stop_words):
    if not text:
        return []
    try:
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(text.lower())
        return [word for word in tokens if word.isalpha() and word not in stop_words]
    except Exception as e:
        print(f"Error during text cleaning: {e}")
        return []

def load_stop_words(stop_words_dir):
    stop_words = set()
    for filename in os.listdir(stop_words_dir):
        try:
            if filename.endswith('.txt'):
                with open(os.path.join(stop_words_dir, filename), 'r', encoding='utf-8', errors='ignore') as file:
                    stop_words.update(set(file.read().split()))
        except Exception as e:
            print(f"⚠️ Error loading {filename}: {e}")
    return frozenset(stop_words) # Just in case for thread safety

def load_master_dictionary(positive_file, negative_file, stop_words):
    positive_words, negative_words = set(), set()  # ✅ FIX ensure always defined
    try:
        with open(positive_file, 'r', encoding='utf-8', errors='ignore') as file:
            positive_words = set(word.strip() for word in file.read().split() if word.strip() not in stop_words)
    except Exception as e:
        print(f"⚠️ Error loading {positive_file}: {e}")    

    try:
        with open(negative_file, 'r', encoding='utf-8', errors='ignore') as file:
            negative_words = set(word.strip() for word in file.read().split() if word.strip() not in stop_words)
    except Exception as e:
        print(f"⚠️ Error loading {negative_file}: {e}")    

    return positive_words, negative_words

@lru_cache(maxsize=10000)
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
        'avg_number_of_words_per_sentence': avg_words_per_sentence,
        'complex_word_count': num_complex_words,
        'word_count': word_count,
        'syllables_per_word': avg_syllables_per_word,
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

@retry(stop=stop_after_attempt(CONFIG['MAX_RETRIES']),
       wait=wait_exponential(multiplier=1, min=2, max=10))
def extract_article(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive'
        }
        response = requests.get(url, headers=headers, timeout=30) #Setting timeout as 30, since some sites are not loading quickly(raised from 10s)
        response.raise_for_status()  # Raise an exception for HTTP errors

        soup = BeautifulSoup(response.text, 'html.parser')
        title_el = soup.find('h1', class_='entry-title')
        content_div = soup.find('div', class_='td-post-content')
        if not title_el or not content_div:
            return None, None
        for element in content_div.find_all(['div', 'script', 'style', 'iframe', 'ins']):
            element.decompose()
        
        article_text = content_div.get_text('\n', strip=True)

        article_text = re.sub(r'\n\s*\n', '\n\n', article_text)
        return title_el.get_text(strip=True), article_text
    except Exception as e:
        print(f"⚠️ Error extracting article: {e} for the url: {url}")
        return None, None

def analyze_article(url, url_id, stop_words, positive_words, negative_words):
    try:
        title, text = extract_article(url)
        if not text:
            return None

        cleaned_tokens = clean_text(text, stop_words)

        if not cleaned_tokens:
            return None
        
        # Calculate sentiment scores
        sentiment_scores = calculate_sentiment_scores(cleaned_tokens, positive_words, negative_words)

        # Calculate readability scores
        readability_scores = calculate_readability_scores(text, cleaned_tokens)
        
        # Combine all results
        results = {
            'url_id': url_id,
            'url': url,
            **sentiment_scores,
            **readability_scores
        }
        
        return results
    except Exception as e:
        print(f"⚠️ Analysis failed for {url_id}: {e}")
        return None
    
def read_urls_from_excel(file_path, url_col='URL', url_id_col='URL_ID'):
    try:
        # Read the Excel file
        df = pd.read_excel(file_path)
        
        # Extract URL and URL_ID columns
        urls = df[url_col].tolist()
        url_ids = df[url_id_col].tolist()
        
        # Return as list of tuples
        return list(zip(url_ids, urls))
    
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return []
    
def save_to_file(results, fileLength):
    try:
        # Saving once at the end
        if results:
            df = pd.DataFrame(results)
            df.rename(columns={
                col: format_metric_name(col) for col in df.columns
                if col not in ['url_id', 'url', 'title']
            }, inplace=True)
            df.to_excel(CONFIG['OUTPUT_FILE'], index=False)
            print(f"\n✅ Processing complete! Successfully processed {len(results)}/{fileLength} URLs")
            print(f"📊 Results saved to {CONFIG['OUTPUT_FILE']}")
        else:
            print("\n⚠️ No results to save.")
    except Exception as e:
        print(f"Error writing Excel file: {e}")
        return []

def main():
    try:
        """Main execution flow"""
        print("🚀 Initializing Text Analyzer")

        #Remove any old output file if exists (Instead of maintaining duplicates)
        if os.path.exists(CONFIG['OUTPUT_FILE']):
            os.remove(CONFIG['OUTPUT_FILE'])
            print(f"🗑️  Removed existing {CONFIG['OUTPUT_FILE']}, starting fresh...")

        stop_words = load_stop_words(CONFIG['STOPWORDS_DIRECTORY'])
        positive_words, negative_words = load_master_dictionary(CONFIG['POSITIVE_DICT'], CONFIG['NEGATIVE_DICT'], stop_words)

        read_data = read_urls_from_excel(CONFIG['INPUT_FILE'])
        if not read_data:
            print("❌ No URLs found in input file")
            return

        print(f" --> Processing {len(read_data)} URLs...")
        results_list = []

        for url_id, url in tqdm(read_data, desc="Processing URLs"):
            time.sleep(CONFIG['REQUEST_DELAY'])
            result = analyze_article(url, url_id, stop_words, positive_words, negative_words)
            if result:
                results_list.append(result)
        
        save_to_file(results_list, len(read_data))
    except Exception as e:
        print(f"Error on the main function: {e}")

    

if __name__ == '__main__':
    main()
