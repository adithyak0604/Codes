import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
response={  "hello":"Hi there! How can I assist you?",
            "weather":"The weather is great today!",
            "name":"I'm ChatGPT's cousin, here to help you!",
            "bye":"Goodbye! Have a fantastic day!"
        }

def nltk_chatbot():
    print("NLTK ChatBot: Hi! Ask me anything. Type 'bye' to exit.\n")
    while True:
        user_input=input("You: ").lower()
        if user_input == "bye":
            print("NLTK ChatBot:",response["bye"])
            break
        tokens=word_tokenize(user_input)
        filtered_words=[word for word in tokens if word not in stopwords.words('english')]
        found=False
        for word in filtered_words:
            if word in response:
                print("NLTK ChatBot:",response[word])
                found=True
                break
        if not found:
            print("NLTK ChatBot: Sorry, I didn't quite get that. Try again!")

nltk_chatbot()