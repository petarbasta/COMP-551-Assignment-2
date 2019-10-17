import csv
import math
import numpy as np
import sklearn as sk
import nltk as nk
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB

allWordCounts = {}
numComments = []
globalVocab = set()

# TODO return a preprocessed dataset and it's values

def parseTrainingData(filename):
    input, output = [], []
    with open(filename, encoding='UTF-8') as file:
        reader = csv.reader(file)
        for row in reader:
            input.append(row[1])
            output.append(row[2])
    input.remove(input[0])
    output.remove(output[0])
    return input, output

def parseTestingData(filename):
    input = []
    with open(filename, encoding='UTF-8') as file:
        reader = csv.reader(file)
        for row in reader:
            input.append(row[1])
    return input

def lemmatize(input, stopWords):
    lemmatizer = nk.stem.WordNetLemmatizer()
    for i, line in enumerate(input):
        words = nk.word_tokenize(line)
        input[i] = ' '.join([lemmatizer.lemmatize(word) for word in words if word not in stopWords])

def stem(input, stopWords):
    stemmer = nk.stem.PorterStemmer()
    for i, line in enumerate(input):
        words = nk.word_tokenize(line)
        input[i] = ' '.join([stemmer.stem(word) for word in words if word not in stopWords])

def preprocess(input, stop_words, s, l):
    if s:
        stem(input, stop_words)
    if l:
        lemmatize(input, stop_words)

def processOutput(input):
    for i, line in enumerate(input):
        if(input[i] == "hockey"):
            input[i] = 0
        if(input[i] == "nba"):
            input[i] = 1
        if (input[i] == "wow"):
            input[i] = 2
        if (input[i] == "leagueoflegends"):
            input[i] = 3
        if (input[i] == "soccer"):
            input[i] = 4
        if (input[i] == "funny"):
            input[i] = 5
        if (input[i] == "anime"):
            input[i] = 6
        if (input[i] =="trees"):
            input[i] = 7
        if (input[i] == "Overwatch"):
            input[i] = 8
        if (input[i] == "nfl"):
            input[i] = 9
        if (input[i] == "GlobalOffensive"):
            input[i] = 10
        if (input[i] == "AskReddit"):
            input[i] = 11
        if (input[i] == "worldnews"):
            input[i] = 12
        if (input[i] == "europe"):
            input[i] = 13
        if (input[i] == "canada"):
            input[i] = 14
        if (input[i] == "gameofthrones"):
            input[i] = 15
        if (input[i] == "Music"):
            input[i] = 16
        if (input[i] == "movies"):
            input[i] = 17
        if (input[i] == "baseball"):
            input[i] = 18
        if (input[i] == "conspiracy"):
            input[i] = 19

def toBigram(comments):
    bigrams = []
    for i, comment in enumerate(comments):
        bigrams.append([])
        for j in range (len(comment)-1):
            bigrams[i].append(comment[j] + '_' + comment[j+1])
    return bigrams

def toTrigram(comments):
    trigrams = []
    for i, comment in enumerate(comments):
        trigrams.append([])
        for j in range (len(comment)-2):
            trigrams[i].append(comment[j] + '_' + comment[j+1] + comment[j+2])
    return trigrams

def toUnigram(input):
    return sk.feature_extraction.text.CountVectorizer().fit_transform(input)

def getDataset():
    totalInput, totalOutput = parseTrainingData("reddit_train.csv")
    preprocess(totalInput, set(stopwords.words('english')), True, True)
    processOutput(totalOutput)
    a = []
    for i in totalInput:
        a.append(str.split(i))

    return a, totalOutput

# returns dictionary with words and counts
def getWordCounts(words):
    wordCounts = {}
    for word in words:
        wordCounts[word] = wordCounts.get(word, 0.0) + 1.0
    return wordCounts

# trains the model
def fit(X, Y):
    # totalNumComments = len(X)

    # for each subreddit
    for i in range (20):
        # count how many comments belong to that subreddit
        numComments.append(sum(1 for label in Y if label == i))
        # calculate what percentage of comments are in that subreddit
        # logClassPercentages.append(math.log(numComments[i] / totalNumComments))

    # for each comment
    for x, y in zip(X, Y):
        # get dictionary of words and their counts
        counts = getWordCounts(x)

        # for each word
        for word, count in counts.items():
            # put in global vocab
            if word not in globalVocab:
                globalVocab.add(word)
            # put in that subreddits vocab with count
            if word not in allWordCounts[y]:
                allWordCounts[y][word] = 0.0

            allWordCounts[y][word] += count

# predicts subreddits for a set of comments
def predict(X):
    print ("here")

    print (len(X))
    results = []
    # for each comment
    for x in X:
        # get dictionary of words and their counts
        counts = getWordCounts(x)
        scores = []
        for i in range (20):
            scores.append(0)
        
        # for each word
        for word, _ in counts.items():
            # if not in global vocab, skip
            if word not in globalVocab: continue
                
            # for each subreddit
            for i in range(20):
                    # calculate score for that word with Laplace smoothing
                    scores[i] += math.log((allWordCounts[i].get(word, 0.0) + 1) / (numComments[i] + 2))

        # for each subreddit
        # for i in range(20):
            # add log of percentage of comments in that subreddit
            # scores[i] += logClassPercentages[i]

        # predict subreddit with highest score
        results.append(scores.index(max(scores)))

    # returns list of predictions for X
    return results

# splits dataset into k lists
def split(dataset, k):
    x, y = divmod(len(dataset), k)
    datasets = []
    for i in range(k):
        datasets.append(dataset[i * x + min(i, y):(i + 1) * x + min(i + 1, y)])
        
    return datasets

# splits dataset into k lists (len() does not work with the unigram)
def splitShape(dataset,k):
    x, y = divmod(dataset.shape[0], k)
    datasets = []
    for i in range(k):
        datasets.append(dataset[i * x + min(i, y):(i + 1) * x + min(i + 1, y)])

    return datasets

# runs k fold cross validation
def kFoldCrossValidation(datasets, values):
    accuracies = []

    # for each dataset
    for i in range(len(datasets)):
        global allWordCounts
        global globalVocab
        for k in range (20):
            allWordCounts[k] = {}
        globalVocab = set()
        trainingSet = []
        trainingSetValues = []
        validationSet = []
        validationSetValues = []

        for j in range(len(datasets)):
            if (i != j):
                # put everything that isn't the current dataset into the training set
                list.extend(trainingSet, datasets[j])
                list.extend(trainingSetValues, values[j])
            else:
                # set the validation set to the current dataset
                validationSet = datasets[j]
                validationSetValues = values[j]

        # train the model
        fit(trainingSet, trainingSetValues)

        # predict the outputs
        predictedValues = predict(validationSet)

        # check how accurate the model is
        accuracies.append(evaluate_acc(validationSetValues, predictedValues))
        print (accuracies)
        
    #return the average of the accuracy
    return np.mean(accuracies)


def LR(trainingInput, trainingOutput, validationInput):
    model = sk.linear_model.LogisticRegression(solver = 'liblinear' , random_state = 35)
    model.fit(trainingInput, trainingOutput)
    return model.predict(validationInput)

def SVM(trainingInput, trainingOutput, validationInput):
    model = sk.svm.LinearSVC(random_state = 35)
    model.fit(trainingInput, trainingOutput)
    return model.predict(validationInput)

def NB(training_input, training_output, validation_input):
    model = MultinomialNB(alpha = 0.85)
    model.fit(training_input, training_output)
    return model.predict(validation_input)

# calculate accuracy of model
def evaluate_acc(validationSetValues, predictedValues):
    return sum(1 for i in range(len(predictedValues)) if predictedValues[i] == validationSetValues[i]) / float(len(predictedValues))

def main():
    # X Should be words in each comment
    # y should be [0,19] for the corresponding subreddit

    X, y = getDataset()
    # X = toBigram(X)
    # X = toTrigram(X)

    #Split into 10 parts
    datasets = split(X, 5)
    values = split(y, 5)

    # #Get accuracy from k-fold
    accuracy = kFoldCrossValidation(datasets, values)

    print(accuracy)

main()



# totalInput, totalOutput = parseTrainingData("reddit_train.csv")
# preprocess(totalInput, set(stopwords.words('english')), True, True)
# trainingInput, validationInput, trainingOutput, validationOutput = \
#     sk.model_selection.train_test_split(toUnigram(totalInput), totalOutput, test_size = 0.15, random_state = 35)
# print(sk.metrics.accuracy_score(validationOutput, LR (trainingInput,trainingOutput,validationInput)))