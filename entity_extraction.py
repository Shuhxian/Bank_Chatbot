import spacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()
from pprint import pprint

def get_entities(sent):
    """Returns all the entities found in the string 'sent' in a dictionary
    
    Doesn't do any prior preprocessing at all
    
    Parameters
    ----------
    sent : str
        string in which all the entities need to be extracted from

    """
    entities_dict = dict([(str(x), x.label_) for x in nlp(sent).ents])
    return entities_dict

if __name__ == '__main__':
    # Code mostly modified from https://towardsdatascience.com/named-entity-recognition-with-nltk-and-spacy-8c4a7d88e7da
    # To test whether if it works just run this file: "py entity_extraction"
    from bs4 import BeautifulSoup
    import requests
    import re

    def url_to_string(url):
        res = requests.get(url)
        html = res.text
        soup = BeautifulSoup(html, 'html5lib')
        for script in soup(["script", "style", 'aside']):
            script.extract()
        return " ".join(re.split(r'[\n\t]+', soup.get_text()))

    ny_bb = url_to_string('https://www.nytimes.com/2018/08/13/us/politics/peter-strzok-fired-fbi.html?hp&action=click&pgtype=Homepage&clickSource=story-heading&module=first-column-region&region=top-news&WT.nav=top-news')
    entities_dict = get_entities(ny_bb)
    pprint(entities_dict)