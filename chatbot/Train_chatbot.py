# In this file, we will build and train the deep learning model
# that can classify and identify what the user is asking to the bot.


# Numpy is a library widely used for working with Multi dimensional Arrays and Matrix
# And perform mathematical operations in this data structures
import numpy as np
# Keras is a Deep Learning library, have high and low level APIs for building and traning models.
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
# Implements pseudo-random number generators
import random
# The Natural Language Toolkit, is a library for symbolic and statistical natural language (NLP)
# Helps with classification, tokenization, stemming, tagging, parsing and semantic reasoning functionalities
import nltk
# nltk.download('all')
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
import json
# Pickling is when a object is converted to a byte stream, and unpicking is the inverse
# The library Pickle will help in the Pickling and unpicking
import pickle

# Open the json file, and saving in a variable
intents_file = open('intents.json').read()
intents = json.loads(intents_file)

# ** Processing the data **
# The raw data in the Json File need to be processed in order to be used.

words = []
classes = []
documents = []
ignore_letters = ['!', '?', ',', '.']

# Tokenizing the data will break the sentences in words add in a list.
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        # Add documents in the corpus
        documents.append((word, intent['tag']))
        # Add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

print(documents)

# Another is ways is through Lemmatization. That convert words into the lemma form
# reducing all the canonical words.

# For example, the words play, playing, plays, played, etc. will all be replaced with play.
# This way, we can reduce the number of total words in our vocabulary.
# So now we lemmatize each word and remove the duplicate words.

# lemmaztize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))
# sort classes
classes = sorted((list(set(classes))))
# documents = combination between patterns and intents
print(len(documents), "documents")
# classes = intents
print(len(classes), "classes", classes)
# words = all words, vocabulary
print(len(words), "unique lemmatized words", words)

# 'words' contain the vocabulary of the chatbot and 'classes' contain the total of entities to  classify

# Step 3. Create Training and Testing Data

# Creating the training data
training = []
# Empty Array for the output
output_empty = [0] * len(classes)
# Training set, bag of words for every sentence
for doc in documents:
    # Initializing bag of words
    bag = []
    # List of tokenized words for the pattern
    word_patterns = doc[0]
    # Lemmatize each word - create base word, in attempt to represent related words
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    # Create the bag of words array with 1, if word is found in current position
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    # Output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Shuffle the features and make numpy array
random.shuffle(training)
training = np.array(training)
# Create training and testing lists. X - Patterns, Y - Intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print('Training data is created')

# Step 4. Training the Model

# Deep neural networds model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0],),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compiling model
# SGD with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Training and saving the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print("model is created")