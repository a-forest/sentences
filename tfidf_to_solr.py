# -*- coding: utf-8 -*-

###Requirements: one solrbook of apache solr , nltk

""""
(solrbook schema)
-----------------
curl -X POST -H 'Content-type:application/jason' --data-binary '{
"add-field":{
  "name":"words",
  "type":"string"
},
"add-field":{
  "name":"values",
  "type":"float",
  "multiValued":"true",
  "indexed":"false",
  "stored":"true"
}
}' http://127.0.0.1:8983/solr/solrbook/schema
"""

import requests
import json
import numpy as np
from nltk.corpus import gutenberg, brown
from sklearn.feature_extraction.text import TfidfVectorizer

###read from corpus and register tfidf_table to solrbook of apache solr
class TfidfToSolr: 
    
    def _read_corpus(self, num_paras):      
        paras = []
        ####brown corpus
        print("reading corpus")
        paras += brown.paras(categories=['romance'])
        paras += brown.paras(categories=['adventure'])
        paras += brown.paras(categories=['fiction'])
        paras += brown.paras(categories=['mystery'])
        paras += brown.paras(categories=['science_fiction'])
        paras += brown.paras(categories=['religion'])
        paras += brown.paras(categories=['humor'])
        paras += brown.paras(categories=['hobbies']) 
        paras += brown.paras(categories=['lore'])
        paras += brown.paras(categories=['learned'])
        paras += brown.paras(categories=['religion'])
        paras += brown.paras(categories=['reviews'])
        paras += brown.paras(categories=['belles_lettres'])
        paras += brown.paras(categories=['editorial'])
        paras += brown.paras(categories=['news'])
        ###gutenberg corpus
        """
        paras +=  gutenberg.paras("austen-emma.txt")
        paras +=  gutenberg.paras("austen-persuasion.txt")
        paras +=  gutenberg.paras("austen-sense.txt")
        paras +=  gutenberg.paras("bible-kjv.txt")
        paras +=  gutenberg.paras("blake-poems.txt")
        paras +=  gutenberg.paras("bryant-stories.txt")
        paras +=  gutenberg.paras("burgess-busterbrown.txt")
        paras +=  gutenberg.paras("carroll-alice.txt")
        paras +=  gutenberg.paras("chesterton-ball.txt")
        paras +=  gutenberg.paras("chesterton-brown.txt")
        paras +=  gutenberg.paras("chesterton-thursday.txt")
        paras +=  gutenberg.paras("edgeworth-parents.txt")
        paras +=  gutenberg.paras("melville-moby_dick.txt")
        paras +=  gutenberg.paras("milton-paradise.txt")
        paras +=  gutenberg.paras("shakespeare-caesar.txt")
        paras +=  gutenberg.paras("shakespeare-hamlet.txt")
        paras +=  gutenberg.paras("shakespeare-macbeth.txt")
        paras +=  gutenberg.paras("whitman-leaves.txt")
        """
        print("paragraph = {}".format(len(paras[0:num_paras])))
        return paras[0:num_paras]

    ###make sentences from the multi dimension array(paras). 
    def _make_raw_list(self, paras):
        print("making raw_sentence_list")
        raw_list = []
        for widesen in paras:
            strline = ''
            for sen in widesen:
                for token in sen:
                    strline = strline + ' ' + str(token.lower())
            raw_list.append(strline)
        print(len(raw_list))
        return raw_list

    ###make tfidf_table using TfidfVectorizer
    def _process_tfidf(self, ngrams, raw_list):
        print("processing tfidf_table")
        tfidf_vect = TfidfVectorizer(ngram_range=(1, ngrams))
        X_tfidf = tfidf_vect.fit_transform(raw_list)
        tfidf_table = X_tfidf.toarray()
    
        wordlist = tfidf_vect.get_feature_names()
        tfidf_table = tfidf_table.T
        print(tfidf_table.shape)
        
        return tfidf_table, wordlist  
    
    
    def __init__(self, solr_url, ngrams, num_paras):
        self.solr_url = solr_url
        self.ngrams = ngrams
        self.num_paras = num_paras
        
        self.paras = self._read_corpus(self.num_paras)
        self.raw_list = self._make_raw_list(self.paras)
        self.tfidf_table, self.wordlist = self._process_tfidf(self.ngrams, self.raw_list)
    
    
    ###register to solrbook(id:words:values)(values is multi dimension vector of tfidf)
    def register_to_solrbook(self):
        print("registering solrbook")
        idindex = 0
        for vector in self.tfidf_table:
            print("{}/{} {}%".format(idindex, len(self.tfidf_table), round(idindex / len(self.tfidf_table) * 100, 3)))
            strvec = "[\""
            string = self.wordlist[idindex]
            index = 0
            for value in vector:
                strvec = strvec + str(round(value, 10)) + "\",\""
            strvec = strvec[:-2] + "]"
            strvec = str(strvec)
            json_string = "{\"add\":{ \"doc\":{\"id\":"+"\"" + str(idindex) + "\",\n\"words\":\"" + string + "\",\n\"values\":" + strvec + "\n},\"boost\":1.0,\"overwrite\":true, \"commitWithin\": 1000}}\n"
            json_data = json.loads(json_string)
            requests.post("{}/update?wt=json".format(self.solr_url), headers={"Content-Type":"application/json"} , data=json_string)
            idindex = idindex + 1
        print("solrbook_register done")


if __name__ == '__main__':
    ###TfidfToSolr('url to the solrbook', number of n-gram, volume of paragraph read from corpus)
    tts = TfidfToSolr('http://localhost:8983/solr/solrbook', 1, 1000)
    tts.register_to_solrbook()
