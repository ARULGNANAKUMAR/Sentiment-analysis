# Import necessary libraries
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import nltk
nltk.download('punkt')  # Download necessary data for TextBlob

# Initialize VADER analyzer
vader_analyzer = SentimentIntensityAnalyzer()

# Function for VADER sentiment analysis
def vader_sentiment(text):
    return vader_analyzer.polarity_scores(text)

# Function for TextBlob sentiment analysis
def textblob_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return {'polarity': polarity, 'label': 'Positive'}
    elif polarity < 0:
        return {'polarity': polarity, 'label': 'Negative'}
    else:
        return {'polarity': polarity, 'label': 'Neutral'}

# Function to analyze sentiment using both VADER and TextBlob
def advanced_sentiment_analysis(text):
    vader_result = vader_sentiment(text)
    textblob_result = textblob_sentiment(text)
    
    print(f"\nText: {text}")
    print("VADER Sentiment Scores:", vader_result)
    print(f"TextBlob Sentiment Polarity: {textblob_result['polarity']}, Label: {textblob_result['label']}")
    
    return vader_result, textblob_result

# Function to analyze a list of sentences and print results
def analyze_sentences(sentences):
    categories = {"Positive": 0, "Negative": 0, "Neutral": 0}

    for sentence in sentences:
        vader_result, textblob_result = advanced_sentiment_analysis(sentence)
        categories[textblob_result['label']] += 1

    print("\nSentiment Summary:")
    print(f"Positive: {categories['Positive']}")
    print(f"Negative: {categories['Negative']}")
    print(f"Neutral: {categories['Neutral']}")

# Example usage
if __name__ == "__main__":
    sentences = []
    
    print("Enter sentences one by one. Type 'done' when finished.")
    
    while True:
        sentence = input("Enter a sentence: ")
        if sentence.lower() == 'done':
            break
        sentences.append(sentence)
    
    if sentences:
        # Perform analysis and print results
        analyze_sentences(sentences)
    else:
        print("No sentences provided for analysis.")
