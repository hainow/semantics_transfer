import nltk
import pickle
from nltk.corpus import wordnet
nltk.download('wordnet')
syndict = {}
antdict = {}

count = 0
for word in wordnet.all_lemma_names():
    count += 1
    if(count % 1000 == 0):
        print(count)
    synonyms = []
    antonyms = []
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name())
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name())
    syndict[word] = synonyms
    antdict[word] = antonyms

with open('synonym.dict', 'wb') as syn_file:
    pickle.dump(syndict,syn_file)

with open('antonym.dict', 'wb') as ant_file:
    pickle.dump(antdict,ant_file)
