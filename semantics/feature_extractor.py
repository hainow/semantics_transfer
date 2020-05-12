"""
Input: a file containing sentences
Output: a feature vector for each sentences, with the indices corresponding
to the semantic feature 

Preprocessing steps
- lowercase all words
- replace common punctuation with no space
- if not in contraction, then replace ' with no space
"""
import time
import sys 
import nltk 
from nltk.corpus import brown
from semantics import CONTRACTIONS, COMMON_PUNCTUATIONS, SLANG_WORDS, \
                    SWEAR_WORDS, EMOTIONAL_WORDS, PROPER_NOUNS, EMOTICONS

# nltk.download('brown')

WORD_SET = set(brown.words())

def is_educated(sentence): 
    # misspelled_count = 0
    for word in sentence: 
        if word not in WORD_SET and word not in PROPER_NOUNS: 
            return False
        #     misspelled_count += 1
        # if misspelled_count > 1: return False 
    return True

def contain_capitalized(sentence): 
    for word in sentence: 
        if word != "I" and word.isupper(): return True
    return False

def is_emotional(preprocessed, original): 
    if (contain_capitalized(original)): 
        return True
    for word in preprocessed: 
        if (word in EMOTICONS or word in EMOTIONAL_WORDS or word.count('!') > 1): 
            return True
    return False

def is_close_relationship(sentence): 
    for word in sentence:
        if word in SLANG_WORDS:
            return True 
    return False

def preprocess(word): 

    # lowercase all word
    word = word.lower()

    # if not in contraction, then replace common punctuation with not space
    if (word not in CONTRACTIONS and word.find("'") != -1): 
        word = word.replace("'", "")
    
    # replace common punctuation with no space 
    for punc in COMMON_PUNCTUATIONS:
        if word.find(punc) != -1: word = word.replace(punc, "")
    
    return word 

def main(): 
    formal_file = sys.argv[1]
    out_file = formal_file + '.semantic_features'
    outputs = [] # List of list eg [[0, 0, 1], [1, 1, 0], ...], where each list is style representation
    i = 0
    start = time.time()
    with open(formal_file, 'r') as f: 
        for line in f.readlines():
            output_line = []
            og = line.strip().split(' ')
            sentence = [preprocess(word) for word in line.strip().split(' ')]
            
            if (is_educated(sentence)): output_line.append("1")    
            else: output_line.append("0")

            if (is_emotional(sentence, og)): output_line.append("0")
            else: output_line.append("1")

            if (is_close_relationship(sentence)): output_line.append("0")
            else: output_line.append("1")

            outputs.append(output_line)

    with open(out_file, 'w') as f: 
        for line in outputs: 
            f.write(" ".join(line) + '\n')

    print("Total time=%f" % (time.time() - start))

if __name__ == "__main__":
   main() 