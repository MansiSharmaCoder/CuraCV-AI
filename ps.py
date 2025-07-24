corpus="""
hello world. welcome in the world.
my name is mansi. my age is 20's, and! i am purcuing BCA! from  
royal educational institute.



"""
#####paragraph ---> sentence###########
#from nltk.tokenize import sent_tokenize

#a=sent_tokenize(corpus)
#print(a)

###########paragraph -----> words ######
#from nltk.tokenize import word_tokenize
#b=word_tokenize(corpus)
#print(b)

#from nltk.tokenize import wordpunct_tokenize
#c=wordpunct_tokenize(corpus)
#print(c)
from nltk.tokenize import TreebankWordTokenizer
#tokenizer=TreebankWordTokenizer()
#d=tokenizer.tokenize(corpus)
d=TreebankWordTokenizer.tokenize(corpus)
print(d)