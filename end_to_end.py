import numpy as np
import pandas as pd
import pickle as pkl
import gzip
import json
import arxiv
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from unidecode import unidecode
import re
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'               #Disabling warnings  
                                                       #https://stackoverflow.com/a/42121886/13036105
from scipy import sparse
import warnings

import keras
from keras_preprocessing.text import tokenizer_from_json                                           
#Loading a tokenizer saved as json. https://stackoverflow.com/questions/45735070/keras-text-preprocessing-saving-tokenizer-object-to-file-for-scoring
                                                       
import tensorflow
from tensorflow.keras.preprocessing.sequence import pad_sequences
import logging
tensorflow.get_logger().setLevel(logging.ERROR)        #Disabling warnings
                                                       #https://stackoverflow.com/a/55142079/13036105

#Importing the classes we defined before in which some of our saved models come.
#https://stackoverflow.com/questions/27732354/unable-to-load-files-using-pickle-and-multiple-modules
from Concat import ConcatModels
from Reduction import clf_reduction
from load_transformer import LoadedTransformer


class math_classifier:
    '''
    Builds an end-to-end classifier based on saved models. 
    The main method is math_classifier.predict which:
        1) based on user's preference, either scrapes math papers (five preprints by default) from arXiv randomly, 
        or receives an arXiv identifier, 
        or a string which should be concatenation of the title and the abstract of a math-related paper;
        2) applies various models to the cleaned text data (which itself is obtained from _text_preprocessing method);
        3) returns predicted probabilities and labels in the form of a dictionary:
        {'3-character MSC':[(14L,0.35)...],'2-character MSC':[(15,0.2),...],'Primary Category':(math.AG,0.77)}.
        
        Caution) The code is written with Mathematics Subject Classification as of 2020 and arXiv identifier scheme as of 2023 in mind.
        Caution) It is on the user to make sure the identifier is valid, otherwise HTTP error may be encountered. 
                    (current identifier format) arXiv:YYMM.number or arXiv:YYMM.numbervV or YYMM.number or YYMM.numbervV    
                    (pre-2007 format) arXiv:<Archive>.<Subject Class>/YYMMnumber or <Archive>.<Subject Class>/YYMMnumber
                                      arXiv:<Archive>.<Subject Class>/YYMMnumbervV or <Archive>.<Subject Class>/YYMMnumbervV
        Caution) The probability for 2-character MSC classes should be interpreted as a lower bound. 
        Caution) The model for the primary arXiv category prediction is trained on data from math and math-physics archives.
        Caution) Two neural network models trained for the primary arXiv category prediction, a convolutional nn and a transformer, the latter is used by default. 
    '''
    #Loading English stop words and some corpus-specific stop words as a class attribute
    __STOPWORDS = list(stopwords.words('english'))+['show','shows','showing','showed','prove','proves','proved','proving','use','uses','using','result','results','resulting','obtain','let',
               'establish','established','establishing','consider','introduce','assume','assuming','denote','denotes','denoting','denoted',
               'describe','investigate','study','discuss','suppose','proof','approach','also','thus','hence','therefore','since','consequently',
              'whose','paper','article','author','authors','via','give','given','gives','iff','could','would','known','certain',
               'moreover','furthermore','although','even though','nevertheless','nonetheless','however', 
               'i.e','e.g','cannot','deduce','demonstrate','demonstrates','exhibit','exhibits','illustrate','illustrates',
               'provide','provided','provides','understand','verify','verifies','verified']
    
    #Loading the stemmer
    __stemmer = SnowballStemmer('english')
    
    
    ################################## START INIT ########################################################
    def __init__(self,
                 MSC_path='./models/MSC_list.json', MSC_simplified_path='./models/MSC_list_simplified.json', 
                 Cat_path='./models/Cat_list.json',
                 clf_MSC_path='./models/clf_MSC.gz', clf_MSC_calibrated_path='./models/clf_MSC_calibrated.gz', 
                 three_to_two_transform_path='./models/3MSC_2MSC.npy',
                 vectorizer_path='./models/vectorizer.gz', tokenizer_path='./models/tokenizer.json',
                 clf_transformer_path='./models/clf_transformer.weights.h5', use_transformer=True,
                 clf_cnn_path='./models/clf_nn.keras'):  
      
        assert all(isinstance(item,str) for item in [MSC_path,MSC_simplified_path,Cat_path,
                                                     clf_MSC_path,clf_MSC_calibrated_path,
                                                     three_to_two_transform_path,
                                                     vectorizer_path,tokenizer_path]), f"[Error] {type(self).__name__} only accepts string inputs which should describe paths."
        assert type(use_transformer)==type(True), "[Error] The use_transformer flag should be Boolean."
        if use_transformer:
            assert isinstance(clf_transformer_path,str), "[Error] A string should be provided as a path to a transformer model."
        else: 
            assert isinstance(clf_cnn_path,str), "[Error] A string should be provided as a path to a convolutional neural net."
    
        #Loading the list of (3-character) MSC classes (variable names for __clf_MSC and __clf_MSC_calibrated)
        assert os.path.exists(MSC_path), f"[Error] {MSC_path} does not exist."
        with open(MSC_path,'r') as file:
            self.__MSC_list=json.load(file)
        
        #Loading the list of (2-character) simplified MSC classes (variable names for __clf_MSC_simplified)
        assert os.path.exists(MSC_simplified_path), f"[Error] {MSC_simplified_path} does not exist."
        with open(MSC_simplified_path,'r') as file:
            self.__MSC_simplified_list=json.load(file)
            
        #Loading the list of categories of math archive (variable names for __clf_nn)
        assert os.path.exists(Cat_path), f"[Error] {Cat_path} does not exist."
        with open(Cat_path,'r') as file:
            self.__Cat_list=json.load(file)
        
        #Loading the trained classifier which outputs MSC classes
        assert os.path.exists(clf_MSC_path), f"[Error] {clf_MSC_path} does not exist."
        with gzip.open(clf_MSC_path,'rb') as file:
            self.__clf_MSC=pkl.load(file)
            
        #Loading the trained classifier which outputs probabilities of MSC classes
        assert os.path.exists(clf_MSC_calibrated_path), f"[Error] {clf_MSC_calibrated_path} does not exist."
        with gzip.open(clf_MSC_calibrated_path,'rb') as file:
            self.__clf_MSC_calibrated=pkl.load(file)
        
        #Loading the transformation mapping (3-character) MSC classes to (2-character) simplified MSC classes
        #And construct the simplified classifier based on it
        assert os.path.exists(three_to_two_transform_path), f"[Error] {three_to_two_transform_path} does not exist."
        self.__T=np.load(three_to_two_transform_path)
        self.__clf_MSC_simplified=clf_reduction(self.__clf_MSC,self.__T,self.__clf_MSC_calibrated)
            
        
        #Loading the vectorizer used to encode the data before training classifiers of MSC labels
        assert os.path.exists(vectorizer_path), f"[Error] {vectorizer_path} does not exist."
        with gzip.open(vectorizer_path,'rb') as file:
            self.__vectorizer=pkl.load(file)
        
        #Loading the tokenizer used to encode the data before training the classifier of primary category
        assert os.path.exists(tokenizer_path), f"[Error] {tokenizer_path} does not exist."
        with open(tokenizer_path) as file:
            self.__tokenizer=tokenizer_from_json(json.load(file))
        
        #Loading one of the two trained neural net classifiers which outputs the probabilities for primary math categories
        if use_transformer:
            assert os.path.exists(clf_transformer_path), f"[Error] {clf_transformer_path} does not exist."
            self.__clf_nn=LoadedTransformer(clf_transformer_path).transformer
        else:
            assert os.path.exists(clf_cnn_path), f"[Error] {clf_cnn_path} does not exist."
            self.__clf_nn=keras.models.load_model(clf_cnn_path)
            
    ################################## END INIT ##########################################################    
    
    
    ################################## The key method ####################################################
    def predict(self,n_random=None,text=None,identifier=None):  
        
        #Sanity check regarding the inputs
        if text is not None:
            flag='text'
            if n_random is not None or identifier is not None:
                warnings.warn("More than one input, only the text input will be considered.")
            if not isinstance(text,str):
                raise ValueError("The text input should be a string.")
        elif identifier is not None: 
            flag='identifier'
            if n_random is not None:
                warnings.warn("More than one input, only the arXiv identifier input will be considered.")
            if not isinstance(identifier,str):
                raise ValueError("The arXiv identifier input should be a string.")
        elif n_random is not None:
            flag='random'
            if not isinstance(n_random,int):
                raise ValueError("The n_random input should be a positive integer.")
            if n_random<=0:
                raise ValueError("The n_random input should be a positive integer.")
            if n_random>20:
                warnings.warn("It is strongly suggested not to use this function for scraping more than 20 papers.")
        else:
            flag='random'
            n_random=5
        
        #Based on flag which determines the input type, we compute the output.
        if flag=='text':
            cleaned_text=self._text_preprocessing(text)     #Preprocessing     
            return self.__predict_from_cleaned_text(cleaned_text)
        
        if flag=='identifier':
            identifier=identifier.removeprefix('arXiv:')    #Remove the prefix if necessary
            client = arxiv.Client()                         #Construct the default API client.
            search = arxiv.Search(id_list=[identifier])
            results = client.results(search)                #This is an iterable.                               
            try:
                paper=next(results)                         #HTTP error may be encountered if the identifier is not valid.
                scraped_text=paper.title+" "+paper.summary
                cleaned_text=self._text_preprocessing(scraped_text)
                return self.__predict_from_cleaned_text(cleaned_text)
            except:                                         #To catch cases where the iterable is empty.
                raise ValueError("The provided identifier didn't match any arXiv preprint, make sure it is valid.")
                
        if flag=='random':    
            client = arxiv.Client()                     #Construct the default API client.
            #We search for the keyword math, and we scrape 10+the number specified by the user.
            #Because we may need to drop some of them which have nothing to do with the math archive.
            search = arxiv.Search(query = "math",max_results = 10+n_random,sort_by = arxiv.SortCriterion.SubmittedDate)
            results = client.results(search)            #This is an iterable.    
            n_outputted=0
            for paper in results:
                if n_outputted>=n_random:
                    break
                if len(set(paper.categories).intersection(set(self.__Cat_list)))==0:
                    continue                            #The scraped paper is probably not math related.
                else:
                    print(f'Paper {n_outputted+1}: {paper.entry_id}')
                    scraped_text=paper.title+" "+paper.summary
                    cleaned_text=self._text_preprocessing(scraped_text)
                    print(self.__predict_from_cleaned_text(cleaned_text),'\n')
                    n_outputted+=1
            return

        
    ################################## Auxiliary #########################################################         
    def __predict_from_cleaned_text(self,cleaned_text):    
        
        '''
        Receives the preprocessed text and returns the labels and their probabilities.
        '''
        
        vectorized=self.__vectorizer.transform([cleaned_text])               #A sparse matrix
        
        if vectorized.sum()==0:
            print('Not enough relevant words detected, make sure that the entered text is the abstract (or the title) of a math-related paper.')
            return None
        
        MSC_label_pred=self.__clf_MSC.predict(vectorized)                   #numpy array of size 1*(the number of MSC classes)
        MSC_proba_pred=self.__clf_MSC_calibrated.predict_proba(vectorized)  #numpy array of size 1*(the number of MSC classes)
        
        MSC_simplified_label_pred=self.__clf_MSC_simplified.predict(vectorized)       #numpy array of size 1*(the number of simplified MSC classes)
        MSC_simplified_proba_pred=self.__clf_MSC_simplified.predict_proba(vectorized) #numpy array of size 1*(the number of simplified MSC classes)
        
        nn_input=self.__padder(self.__tokenizer.texts_to_sequences([cleaned_text])) #What the neural net receives as input. 
        Cat_proba_pred=self.__clf_nn.predict(nn_input,verbose=0)                    #numpy array of size 1*(the number of math-related primary categories)    
        
        return self.__build_output(MSC_label_pred,MSC_proba_pred,
                MSC_simplified_label_pred,MSC_simplified_proba_pred,
                Cat_proba_pred)

        
    #For padding sequences with parameter as set for training the neural network (the Archive Primary Category Prediction notebook)
    @staticmethod
    def __padder(X):  
        return pad_sequences(X, maxlen=100, padding='post')
        
    
    def __build_output(self,
                      MSC_label_pred,MSC_proba_pred,
                      MSC_simplified_label_pred,MSC_simplified_proba_pred,
                      Cat_proba_pred):
        '''
        Receives the predicted labels and probabilities and constructs an output of the form 
        {'3-character MSC':[(14L,0.35)...],'2-character MSC':[(15,0.2),...],'Primary Category':(math.AG,0.77)}
        consisting of predicted labels and their associated probabilities obtained from
        classifiers of 3-character and 2-character MSC classes (multi-label tasks), 
        and from the classifier of the primary arXiv category (a multi-class task). 
        (In the former two, when no class is predicted, a single class with the highest probability will be picked.)
        '''
        #Initializing
        output_dict={}
        
        #For the 3-character MSC prediction task (multi-label)
        output_dict['3-character MSC']=[]
        indices=np.unique(np.where(MSC_label_pred[0]==1)).tolist()
        if len(indices)==0:                                    #The case where the multi-label classifiers returns no label.
            indices=[np.argmax(MSC_proba_pred[0])]
        for index in indices:
            output_dict['3-character MSC']+=[(self.__MSC_list[index],round(MSC_proba_pred[0][index],2))]
            
        #For the 2-character MSC prediction task (multi-label)
        output_dict['2-character MSC']=[]
        indices=np.unique(np.where(MSC_simplified_label_pred[0]==1)).tolist()
        if len(indices)==0:                                    #The case where the multi-label classifiers returns no label.
            indices=[np.argmax(MSC_simplified_proba_pred[0])]
        for index in indices:
            output_dict['2-character MSC']+=[(self.__MSC_simplified_list[index],round(MSC_simplified_proba_pred[0][index],2))]
            
        #For the primary category prediction task (multi-class)
        index=np.argmax(Cat_proba_pred[0])
        output_dict['Primary Category']=(self.__Cat_list[index],round(Cat_proba_pred[0][index],2))
        
        return output_dict
         
       
    ################################ Text preprocessor and its smaller constituent functions #############
    @classmethod
    def _text_preprocessing(cls,string):
        string=cls.__remove_math(string)
        string=cls.__remove_link(string)
        string=cls.__make_lower(string)
        string=cls.__kill_accent(string)
        string=cls.__modify(string)
        string=cls.__remove_stop_words(string,cls.__STOPWORDS)
        string=cls.__stemming(string,cls.__stemmer)
        string=cls.__special_character_removal(string)
        string=cls.__final_polish(string)
        return string
        
    @staticmethod
    def __remove_math(string):
        terms=string.split('$')
        if terms[0]=='' and terms[-1]=='' and len(terms)==3:   #I.e. the whole text is in the math environment. 
            return ''

        cleaned_string=''                  #Initializing
        between_dollar_signs=False        
        for term in terms:
            if not between_dollar_signs:
                cleaned_string+=term
                between_dollar_signs=True  #The next term will be between dollar signs
            else:
                between_dollar_signs=False
        return cleaned_string

    @staticmethod
    def __remove_link(string):
        cleaned_string=''
        terms=string.split(' ')
        for term in terms:
            if term.startswith('http') or term.startswith('arXiv'):
                continue
            cleaned_string+=' '+term
        cleaned_string=cleaned_string.removeprefix(' ')  #Killing any space at the beginning of the string
        return cleaned_string
    
    @staticmethod
    def __make_lower(string):
        return string.lower()
    
    @staticmethod
    def __kill_accent(string):
        return unidecode(string)

    @staticmethod
    def __modify(string):
    
        #The TeX commands used for italic will be removed 
        string=string.replace('\\emph',' ')
        string=string.replace('\\textit',' ')

        #The following characters will be removed.
        string=string.translate({ord(i): None for i in ":;,!?.'(){}`[]/\*^$"})
        string=string.translate({ord(i): None for i in '"'})     #The single quote was removed above, now the double quote.


        #The following characters will be replaced with a space.
        string=string.translate({ord(i): ' ' for i in '\n'})     #Caution: Sometimes, there is no space after \n.                           
        string=string.translate({ord(i): ' ' for i in "-=&+"}) 

        return string
    
    @staticmethod
    def __remove_stop_words(string,STOPWORDS):
        cleaned_string=''
        terms=string.split(' ')
        for term in terms:
            if (term not in STOPWORDS) and len(term)>0:  #Avoiding empty string '' which comes up in terms if there are consecutive spaces. 
                cleaned_string+=' '+term
        cleaned_string=cleaned_string.removeprefix(' ')  #Killing any space at the beginning of the string
        return cleaned_string
    
    @staticmethod
    def __stemming(string,stemmer):
        cleaned_string=''
        terms=string.split(' ')
        for term in terms:
                cleaned_string+=' '+stemmer.stem(term)
        cleaned_string=cleaned_string.removeprefix(' ')  #Killing any space at the beginning of the string
        return cleaned_string
    
    @staticmethod
    def __special_character_removal(string):
        cleaned_string=''
        terms=string.split(' ')
        for term in terms:
            if len(term)==0 or not term.isalnum():       #Skipping empty string '' or terms with non-alphanumeric characters.
                continue
            cleaned_string+=' '+term
        cleaned_string=cleaned_string.removeprefix(' ')  #Killing any space at the beginning of the string
        return cleaned_string
    
    @staticmethod
    def __final_polish(string):
        string=re.sub(' {2,}',' ',string)
        string.strip(' ')
        return string
    
    ###################################################################################################### 