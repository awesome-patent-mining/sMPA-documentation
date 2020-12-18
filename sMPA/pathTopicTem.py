#coding=UTF-8
"""
This code provides a series of basic functions for topic analysis of main path
"""

# Author: Liang Chen <squirrel_d@126.com>
# License: BSD 3 clause
from gensim import models
from sMPA.semanticRoadMap import *
class PathTopic:
    def __init__(self, texts_serialAddress, pns_serialAddress, lsi_serializationAddress, dictionary_serialAddress, topic_num = 10):
        '''
        init PathTopic class

        Args:
            texts_serialAddress(str): absolute path of texts serialization file
            pns_serialAddress(str): absolute path of patent NO file
            lsi_serializationAddress(str): absolute path of LSI model file
            dictionary_serialAddress(str): absolute path of dictionary file
            topic_num(str): topic number of LSI

        Returns:
            None:
        '''
        with open(texts_serialAddress, 'r') as f:
            self.texts = pickle.load(f)
        with open(pns_serialAddress, 'r') as f:
            self.pns = pickle.load(f)
        with open(lsi_serializationAddress, 'r') as f:
            self.index = pickle.load(f)
        with open(dictionary_serialAddress, 'r') as f:
            self.dictionary = pickle.load(f)
        corpus = [self.dictionary.doc2bow(text) for text in self.texts]
        tfidf = models.TfidfModel(corpus)
        corpus_tfidf = tfidf[corpus]
        self.topic_number = topic_num;
        self.corpus_lsi_np = self.transformCorpusLsi2Numpy(self.index[corpus_tfidf])
