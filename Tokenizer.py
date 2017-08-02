import re

from bs4 import BeautifulSoup

class Tokenizer(object):

    @staticmethod
    def readStopWord():
        stopword_dic = []
        f = open("resources/stopword.dic", 'r')
        while True:
            line = f.readline().strip().strip('\n')
            if not line: break
            if (line[0] == '#'): continue;

            stopword_dic.append(line)
        f.close()
        return stopword_dic

    @staticmethod
    def review_to_words(review, stopwords):
        review_text = BeautifulSoup(review, "html.parser").get_text()
        review_text = re.sub("[^a-zA-Z]", " ", review_text)
        words = review_text.lower().split()

        words = [w for w in words if not w in stopwords]

        return words