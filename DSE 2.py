# Import necessary libraries
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import matplotlib.pyplot as plt
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
    
    print(f"Text: {text}")
    print("VADER Sentiment Scores:", vader_result)
    print(f"TextBlob Sentiment Polarity: {textblob_result['polarity']}, Label: {textblob_result['label']}")
    
    return vader_result, textblob_result

# Function to analyze a list of sentences and visualize results
def analyze_and_visualize(sentences):
    vader_scores = []
    textblob_scores = []
    categories = {"Positive": 0, "Negative": 0, "Neutral": 0}

    for sentence in sentences:
        vader_result, textblob_result = advanced_sentiment_analysis(sentence)
        vader_scores.append(vader_result['compound'])
        textblob_scores.append(textblob_result['polarity'])
        categories[textblob_result['label']] += 1
    
    # Plot results using Matplotlib
    plot_sentiment(vader_scores, textblob_scores, categories)

# Function to plot the sentiment results
def plot_sentiment(vader_scores, textblob_scores, categories):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # VADER and TextBlob compound/polarity scores plot
    axs[0].plot(vader_scores, label="VADER Compound Score", marker='o', linestyle='-', color='b')
    axs[0].plot(textblob_scores, label="TextBlob Polarity Score", marker='x', linestyle='--', color='g')
    axs[0].set_title("Sentiment Scores Comparison")
    axs[0].set_xlabel("Sentence Index")
    axs[0].set_ylabel("Sentiment Score")
    axs[0].legend()
    axs[0].grid(True)

    # Bar chart of sentiment categories
    labels = list(categories.keys())
    values = list(categories.values())
    axs[1].bar(labels, values, color=['green', 'red', 'gray'])
    axs[1].set_title("Sentiment Distribution (TextBlob)")
    axs[1].set_ylabel("Count")
    
    plt.tight_layout()
    plt.show()

sentences = []
    
print("Enter sentences one by one. Type 'done' when finished.")
    
while True:
    sentence = input("Enter a sentence: ")
    if sentence.lower() == 'done':
        break
    sentences.append(sentence)
    
if sentences:
    # Perform analysis and visualize
    analyze_and_visualize(sentences)
else:
    print("No sentences provided for analysis.")
