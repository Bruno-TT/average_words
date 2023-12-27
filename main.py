import numpy as np
import time
from sklearn.decomposition import PCA

PCA_enabled=False # enable PCA, exploratory but generally doesn't help
n=100 # if PCA enabled, how many dimensions to project down to (from 300)

num_closest=100
max_words=60000 # -1 to disable

def load_glove_model(file_path):
    i=0
    glove_model = {}
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array([float(val) for val in split_line[1:]])
            glove_model[word] = embedding
            i+=1
            if i==max_words:
                break
    return glove_model

def PCA_glove_model(glove_model, n):
    global pca
    # Extract the words and their corresponding vectors
    words = list(glove_model.keys())
    vectors = np.array([glove_model[word] for word in words])

    # Perform PCA to reduce the dimensions
    pca = PCA(n_components=n)
    reduced_vectors = pca.fit_transform(vectors)

    # Reconstruct the GloVe model with reduced dimensions
    reduced_glove_model = {word: reduced_vectors[i] for i, word in enumerate(words)}

    return reduced_glove_model, pca

def find_closest_words(vector1, vector2, glove_model, distance_func, variance_ratios=None, num_closest=10):
    closest_words = []
    distances = []

    for word, embedding in glove_model.items():
        distance = distance_func(vector1, vector2, embedding, variance_ratios)
        if len(closest_words) < num_closest:
            closest_words.append(word)
            distances.append(distance)
        else:
            max_distance = max(distances)
            if distance < max_distance:
                max_index = distances.index(max_distance)
                distances[max_index] = distance
                closest_words[max_index] = word

    # Sort the words by their distances
    sorted_indices = sorted(range(len(distances)), key=lambda i: distances[i])
    sorted_words = [closest_words[i] for i in sorted_indices]
    sorted_distances = [distances[i] for i in sorted_indices]

    return sorted_words, sorted_distances

start=time.process_time()

# Load GloVe model
file_path = 'glove.42B.300d.txt'
glove_model = load_glove_model(file_path)

if PCA_enabled:    
    glove_model, current_pca = PCA_glove_model(glove_model, n)
    variance_ratios=current_pca.explained_variance_ratio_

end=time.process_time()

print(f"loaded in {end-start} seconds. {"PCA enabled with n="+str(n) if PCA_enabled else "PCA disabled"}")

from potential_distance_measures import methods

while 1:

    # Input words
    word1 = input("\n\nInput word 1: ")
    word2 = input("Input word 2: ")

    # Retrieve vectors
    w1v = glove_model.get(word1)
    w2v = glove_model.get(word2)

    for distance_func in methods:

        if PCA_enabled:
            closest_words, scores = find_closest_words(w1v, w2v, glove_model, distance_func=distance_func, variance_ratios=variance_ratios, num_closest=20)
        else:
            closest_words, scores = find_closest_words(w1v, w2v, glove_model, distance_func=distance_func, num_closest=num_closest)

        print(f"\n\nUsing Method: {distance_func.__name__}")

        for i, (word, score) in enumerate(zip(closest_words, scores)):
            print(f"{i+1}.\t{word.upper()}\t{score}")


    
