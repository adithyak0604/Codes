from nltk.sentiment import SentimentIntensityAnalyzer
sentiment_analyzer=SentimentIntensityAnalyzer()
sentence="Mumbai Indians has won 5 IPL Trophies."
score=sentiment_analyzer.polarity_scores(sentence)
print("Sentiment Score:",score)