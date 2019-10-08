import math
import numpy as np

wordCounts = {}
numComments = []
logClassPercentages = {}
globalVocab = set()

# TODO return a preprocessed dataset and it's values
def getDataset():
    dataset = [[]]
    return dataset

# returns dictionary with words and counts
def getWordCounts(words):
        for word in words:
            wordCounts[word] = wordCounts.get(word, 0.0) + 1.0
        return wordCounts

# trains the model
def fit(X, Y):

    totalNumComments = len(X)

    # for each subreddit
    for i in range (20):
        # count how many comments belong to that subreddit
        numComments[i] = sum(1 for label in Y if label == i)
        # calculate what percentage of comments are in that subreddit
        logClassPercentages[i] = math.log(numComments[i] / totalNumComments)
        wordCounts[i] = {}

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
            if word not in wordCounts[y]:
                wordCounts[y][word] = 0.0

            wordCounts[y][word] += count

# predicts subreddits for a set of comments
def predict(X):
    results = []
    # for each comment
    for x in X:
        # get dictionary of words and their counts
        counts = getWordCounts(x)
        scores = []
        
        # for each word
        for word, _ in counts.items():
            # if not in global vocab, skip
            if word not in globalVocab: continue
                
            # for each subreddit
            for i in range(20):
                    # calculate score for that word with Laplace smoothing
                    scores[i] += math.log((wordCounts[i].get(word, 0.0) + 1) / (numComments[i] + 2))

        # for each subreddit
        for i in range(20):
            # add log of percentage of comments in that subreddit
            scores[i] += logClassPercentages[i]

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

# runs k fold cross validation
def kFoldCrossValidation(datasets, values):
    accuracies = []

    # for each dataset
    for i in range(len(datasets)):
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
        
    #return the average of the accuracy
    return np.mean(accuracies)

# calculate accuracy of model
def evaluate_acc(validationSetValues, predictedValues):
    return sum(1 for i in range(len(predictedValues)) if predictedValues[i] == validationSetValues[i]) / float(len(predictedValues))

def main():
    # X Should be words in each comment
    # y should be [0,19] for the corresponding subreddit
    X, y = getDataset()

    #Split into 10 parts
    datasets = split(X, 10)
    values = split(y, 10)

    #Get accuracy from k-fold
    accuracy = kFoldCrossValidation(datasets, values)

    print(accuracy)

main()