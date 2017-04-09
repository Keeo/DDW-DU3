import nltk
import wikipedia
from nltk.corpus import words
from nltk.corpus import brown
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.util import *
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter
from string import punctuation
from nltk.corpus import stopwords
stops = stopwords.words('english')
             
text = None
with open('text.txt', 'r') as f:
    text = f.read()
 
# sentence splitting
sentences = nltk.sent_tokenize(text)
 
# tokenization
tokensForSentence = [nltk.word_tokenize(sent) for sent in sentences]
rawTokens = nltk.word_tokenize(text)
tokens = [token for token in rawTokens if token not in punctuation if token not in stops]

# stemming and lemmatization  
stemmer = PorterStemmer()
stems = {token:stemmer.stem(token) for token in tokens}

lemmatizer = WordNetLemmatizer()
lemmas = {token:lemmatizer.lemmatize(token) for token in tokens}

# part of speech tagging
taggedForSentence = [nltk.pos_tag(sent) for sent in tokensForSentence]
tagged = nltk.pos_tag(tokens)
 
# entity recognition
def extractEntities(ne_chunked):
    data = {}
    for entity in ne_chunked:
        if isinstance(entity, nltk.tree.Tree):
            text = " ".join([word for word, tag in entity.leaves()])
            ent = entity.label()
            data[text] = ent
        else:
            continue
    return data
 
ne_chunked = nltk.ne_chunk(tagged, binary=True)


chunked = extractEntities(ne_chunked)

## results
def tokenCounts(tokens):
    counts = Counter(tokens)
    sortedCounts = sorted(counts.items(), key=lambda count:count[1], reverse=True)
    return sortedCounts
    

# top nouns/verbs
def wordType(tagged, ff):
  temp = list()
  for tupple in tagged:
    if (tupple[1] in ff):
      temp.append(tupple[0])
  return temp

print("## nouns/verbs ##")
print("- Top nouns: ")
print(tokenCounts(wordType(tagged, ["NN"]))[0:5])
print("- Top verbs: ")
print(tokenCounts(wordType(tagged, ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]))[0:5])
print()


def entityCount(chunked):
  temp = list()
  for entity in chunked:
    if (type(entity) is tuple):
      temp.append(entity[1])
    else:
      temp.append(entity.label())
  return temp
  
def extractEntitiesWithCount(ne_chunked):
  temp = list()
  for entity in ne_chunked:
    if isinstance(entity, nltk.tree.Tree) and entity.label() == "NE":
      text = " ".join([word for word, tag in entity.leaves()])
      temp.append(text)
  return temp
  
def extractEntitiesTouple(chunked):
  temp = list()
  for entity in chunked:
    if isinstance(entity, tuple):
      temp.append(entity[0])
  return temp
  
print("## POS tagging ##")
print("- Top entities: ")
print(tokenCounts(extractEntitiesTouple(ne_chunked))[0:5])

print("- Top NE entities: ")
print(tokenCounts(extractEntitiesWithCount(ne_chunked))[0:5])

print("- Top types: ")
print(tokenCounts(entityCount(ne_chunked))[0:5])
print()

print("## - NER ##")
print("- NER:")
print(tokenCounts(chunked)[0:5])
print()
  
print("- NER CUSTOM:")
def customPattern(text):
  tokens = nltk.word_tokenize(text)
  tagged = nltk.pos_tag(tokens)
 
  ret = []
  for (i, k) in zip(tagged[:-1], tagged[1:]):
    if (i[1].startswith('JJ') and k[1].startswith("NN")):
      ret.append(i[0] + ' ' + k[0])
        
  return ret

print(tokenCounts(customPattern(text))[0:5])
print();

print("- Custom entity clasfification on wikipedia:")
def magicSentence(sentence):
  tokens = nltk.word_tokenize(sentence)
  tagged = nltk.pos_tag(tokens)

  cache = []
  for entity in tagged:
    if (len(cache) is 0 and not entity[1].startswith("VBZ")):
      continue
    
    if (len(cache) is 0 and entity[1].startswith("VBZ")):
      cache.append(entity)
      continue
      
    if (len(cache) == 1 and entity[1].startswith("DT")):
      cache.append(entity)
      continue
    
    if (len(cache) > 0 and entity[1].startswith("JJ") and cache[-1][1] in ["DT", "VBZ","JJ"]):
      cache.append(entity)
      continue
      
    if (len(cache) > 0 and entity[1].startswith("NN") and cache[-1][1]):
      cache.append(entity)
      break
    
  string = ""
  for entity in cache:
    string += " " + entity[0] 
  return string
  
  
def customEntityClasiffication(text):
  tokens = nltk.word_tokenize(text)
  tagged = nltk.pos_tag(tokens)
  ne_chunked = nltk.ne_chunk(tagged, binary=False)
  extractedEntities = extractEntities(ne_chunked)

  for entity in extractedEntities:
    results = wikipedia.search(entity, 1)
   
    if (len(results) > 0):
      try:
        page = wikipedia.page(results[0])
        sentences = nltk.sent_tokenize(page.summary)
        for sentence in sentences:
          temp = magicSentence(sentence)
          if (temp):
            print(entity + temp)
            break
      except wikipedia.exceptions.DisambiguationError:
        print(entity + " is a thing.")
    else:
      print(entity + " is a thing.")
  
customEntityClasiffication(text)
