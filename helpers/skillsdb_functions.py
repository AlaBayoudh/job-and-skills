
import psycopg2
from os import environ
import pandas as pd
from notebook.services.config import ConfigManager
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk import FreqDist
from nltk.stem.snowball import FrenchStemmer
import operator
from operator import itemgetter, attrgetter
import ast
from langdetect import detect
from unidecode import unidecode
import re
import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")
### Redshift connection
#conn = psycopg2.connect(database=environ['RED_DBNAME'], user=environ['RED_USER'], password=environ['RED_PWD'],host =environ['RED_HOST'],port =environ['RED_PORT']  )
#### Stop words english and french
STOP_WORDS = set(stopwords.words("french")).union(set(stopwords.words("english")))
STOP_WORDS.add('stagiaire')
STOP_WORDS.add('stage')
STOP_WORDS.add('alternance')
STOP_WORDS.add('alternant')
STOP_WORDS.add('apprentissage')
STOP_WORDS.add('offre')
STOP_WORDS.add('embauche')
STOP_WORDS.add('apprenti')
STOP_WORDS.add("junior")
STOP_WORDS.add('cdi')
STOP_WORDS.add('bac+5')
STOP_WORDS.add('bac+4')
STOP_WORDS.add('bac+3')
STOP_WORDS.add('bac+2')
STOP_WORDS.add('intern')
STOP_WORDS.add('bac +')
#### French stemmer
stemmer = FrenchStemmer()
def remove_stop_words(desc):
    """This function removes stop words,special characters and tokenizes a text into
    a list of words and stem them
    
    Arguments:
        desc {[str]} -- [Can be a job offer description,job titles or a skill]
    
    Returns:
        [list] -- [List of words]
    """
    new_desc = re.sub(r'(\S*\d+\S*|[\W_]+|\b([a-z]){1,2}\b)'," ", desc)
    listofwords = word_tokenize(new_desc)
    return [stemmer.stem(m) for m in listofwords if m not in STOP_WORDS]

def load_information_schema(conn):
    """Select tables information schema from Redshift
    
    Arguments:
        conn {[conn]} -- [Redshift connection]
    
    Returns:
        [pd.Dataframe] -- [Data frame]
    """
    return pd.read_sql("select * from information_schema.tables",conn)
#### Offres JT
def load_offres_jt(conn):
    """This function returns a dataframe of JT job offers 
    
    Arguments:
        conn {[conn]} -- [Redshift connection]
    
    Returns:
        [pd.Dataframe] -- [Dataframe with columns (title : offer titles,
                                                   description: offer description
                                                   position_category_id : the offer function "138 in total")]
    """
    return pd.read_sql("select title,description,position_category_id from jt.job_offers",conn) 

def load_offres_jt_2017(conn):
    """This function returns a dataframe of JT job offers from 2017
    
    Arguments:
        conn {[conn]} -- [Redshift connection]
    
    Returns:
        [pd.Dataframe] -- [Dataframe with columns (title : offer titles,
                                                   description: offer description
                                                   position_category_id : the offer function "138 in total")]
    """
    return pd.read_sql("select title,description,position_category_id from jt.job_offers where created_at > '2017-01-01 00:00:00'",conn)  

def tokenize_words_Sents(Sent):
    """Splits Text into words and sentences
    
    Arguments:
        Sent {[str]} -- [Text sample]
    
    Returns:
        [list,list] -- [List of words,list of sentences]
    """
    return word_tokenize(Sent),sent_tokenize(Sent) 

def RegExpTokenizer(Sent):
    """Splits Text into words, also deletes all special characters 
    
    Arguments:
        Sent {[str]} -- [Text sample]
    
    Returns:
        [list] -- [List of words]
    """
    tokenizer = RegexpTokenizer(r'\W+|\d+\S*|\S*\d+')
    return tokenizer.tokenize(Sent)

def eliminate_stop_words(Sent):
    """Deletes Stop words
    
    Arguments:
        Sent {[list]} -- [list of words]
    
    Returns:
        [list] -- [List of words without stop words]
    """
    filtered_words = []
    for w in Sent:
        if w not in STOP_WORDS:
            filtered_words.append(w)
    return filtered_words


def eliminate_irrelevent_Words(ListWords):
    """deletes irrelevent words, in this case:
        - Words with less than 3 characters
        - Digits
    
    Arguments:
        ListWords {[list]} -- [List of words]
    
    Returns:
        [list] -- [List of words after filter]
    """
    NewList = []
    for m in ListWords:
        if m.isdigit() == True:
            pass
        elif len(m) <=3:
            pass
        else:
            NewList.append(m)
    return NewList

#### Bag of words functions
def bag_of_words(ListWords):
    """Bag of words function calculates the frequence of words in a Text
    
    Arguments:
        ListWords {[list]} -- [This is a list of list of Words]
        Ex : [[W1,W2],[W3,W4]]
    
    Returns:
        [Nltk.FreqDist] -- [Returns a dictionary where words are keys and their frequencies as values]
    """
    all_words = []
    for m in ListWords:
        all_words.extend(m)
    all_words = FreqDist(all_words)
    return all_words

def bag_of_bigrams(ListWords):
    """Bag of words function calculates the frequency of Two-words (bi-grams) sequences in a Text
    
    Arguments:
        ListWords {[list]} -- [This is a list of list of Words]
        Ex : [[W1,W2],[W3,W4]]
    
    Returns:
        [Nltk.FreqDist] -- [Returns a dictionary where bi-grams are keys and their frequencies as values]
    """
    all_bigrams = []
    for m in ListWords:
        all_bigrams.extend(list(nltk.bigrams(m)))
    all_bigrams = FreqDist(all_bigrams)
    return all_bigrams

def bag_of_trigrams(ListWords):
    """Bag of words function calculates the frequency of Three-words (Tri-grams) sequences in a Text
    
    Arguments:
        ListWords {[list]} -- [This is a list of list of Words]
        Ex : [[W1,W2],[W3,W4]]
    
    Returns:
        [Nltk.FreqDist] -- [Returns a dictionary where Tri-grams are keys and their frequencies as values]
    """
    new_trigrams = []
    
    for m in ListWords:
        c = 0
        while c < len(m) - 2:
            new_trigrams.append((m[c], m[c+1], m[c+2]))
            c += 1
    new_trigrams = FreqDist(new_trigrams)
    return new_trigrams

def bag_of_fourgrams(ListWords):
    """Bag of words function calculates the frequency of Four-words (Four-grams) sequences in a Text
    
    Arguments:
        ListWords {[list]} -- [This is a list of list of Words]
        Ex : [[W1,W2],[W3,W4]]
    
    Returns:
        [Nltk.FreqDist] -- [Returns a dictionary where Four-grams are keys and their frequencies as values]
    """
    new_fourgrams = []
    
    for m in ListWords:
        c = 0
        while c < len(m) - 3:
            new_fourgrams.append((m[c], m[c+1], m[c+2],m[c+3]))
            c += 1
    new_fourgrams = FreqDist(new_fourgrams)
    return new_fourgrams

def bag_of_fivegrams(ListWords):
    """Bag of words function calculates the frequency of Five-words (Four-grams) sequences in a Text
    
    Arguments:
        ListWords {[list]} -- [This is a list of list of Words]
        Ex : [[W1,W2],[W3,W4]]
    
    Returns:
        [Nltk.FreqDist] -- [Returns a dictionary where Five-grams are keys and their frequencies as values]
    """
    new_fourgrams = []
    
    for m in ListWords:
        c = 0
        while c < len(m) - 4:
            new_fourgrams.append((m[c], m[c+1], m[c+2],m[c+3],m[c+4]))
            c += 1
    new_fourgrams = FreqDist(new_fourgrams)
    return new_fourgrams

def htmlparser(desc):
    """Html parser and changes urls with spaces
    
    Arguments:
        desc {[srt]} -- [Text sample]
    
    Returns:
        [str] -- [Text parsed]
    """
    new_desc = BeautifulSoup(desc,"html.parser").get_text()
    return re.sub(r'(http|www|\S*@)\S*'," ",new_desc)


def preprocessing_text(text):
    """Process text combines three function
       - htmlparser
       - unidecode
       - remove_stop_words
    
    Arguments:
        text {[str]} -- [Text to process]
    
    Returns:
        [str] -- [Text processed]
    """
    bs4_text = htmlparser(text).lower()
    unid_text = unidecode(bs4_text)
    clean_text = remove_stop_words(unid_text)
    return " ".join(clean_text)

def detect_lang(desc):
    """This function detects the language of a text with the help of nltk

    
    Arguments:
        desc {[str]} -- [Text sample]
    
    Returns:
        [str] -- [language]
    """
    try:
        return detect(desc)
    except Exception as e:
        return "Und"

def is_empty(any_structure):
    """Check if any structure is empty
    
    Arguments:
        any_structure {[Any type]} -- [Generaly collections]
    
    Returns:
        [Boolean] -- [description]
    """
    if any_structure:
        return False
    else:
        return True
#### Matching funcition   
def matching(jt_title,altLabels,DF_join_Occ_fr_en):
    """This function matches a processed JT offer title with
    an ESCO job, used with apply() function over a DataFrame of JT offers
    Ex: jt_title = "web developer intern"
        altLabels = [['graphiste','web developer','job3']['job4','job5']]
        'web developer' matched
    Arguments:
        jt_title {[str]} -- [JT Job offer title]
        altLabels {[list]} -- [This is the list of altLabels processed and collected from ESCO, it's a list of list]
        DF_join_Occ_fr_en {[pd.DataFrame]} -- [DataFrame containing 2900 ESCO job title, was used to get the preferred labels of the job]
    
    Returns:
        [list] -- [List of ESCO Jobs that matched with the JT job offer title]
    """
    lista = []
    for listofapp in altLabels:
        for j in listofapp:
            if j in jt_title:
                if len(listofapp[listofapp.index(j)].split(" "))>1:
                    lista.append(DF_join_Occ_fr_en.loc[altLabels.index(listofapp),'preferredLabel_y'])
    return lista

def ast_string_to_list(listOfStrings):
    """This function takes a pandas dataframe column to transfrom string such as '['job']' to lists ['job']
    
    Arguments:
        listOfStrings {[list]} -- [A Dataframe column containg list of values in string format]
    
    Returns:
        [list] -- [A Dataframe column containg list of values in list format]
    """
    return [ast.literal_eval(m) for m in listOfStrings]

def retrieve_final_skills():
    """This function returns the ESCO list of processed skills in french
    
    Returns:
        [pd.Dataframe] -- [Skills dataframe]
    """
    Skills = pd.read_csv("data/all_skills_french_processed.csv")
    Skills.loc[~Skills['Translated_alt_html'].isna(),'Translated_alt_html'] = ast_string_to_list(Skills.loc[~Skills['Translated_alt_html'].isna(),'Translated_alt_html'])
    Skills.loc[~Skills['Translated_alt'].isna(),'Translated_alt'] = ast_string_to_list(Skills.loc[~Skills['Translated_alt'].isna(),'Translated_alt'])
    Skills.loc[~Skills['altLabels_x'].isna(),'altLabels_x'] = ast_string_to_list(Skills.loc[~Skills['altLabels_x'].isna(),'altLabels_x'])
    Skills.loc[~Skills['altLabels_y'].isna(),'altLabels_y'] = ast_string_to_list(Skills.loc[~Skills['altLabels_y'].isna(),'altLabels_y'])
    Skills['Skills_alt_labels'] = ast_string_to_list(Skills['Skills_alt_labels'])
    Skills['French_only_alt'] = ast_string_to_list(Skills['French_only_alt'])
    Skills['French_only_alt_lower_1'] = ast_string_to_list(Skills['French_only_alt_lower_1'])


    return Skills

def retrieve_jt_offers():
    """Returns a dataframe with JT offers from 2017
       and ESCO job that has been matched
    
    Returns:
        [pd.Dataframe] -- [JT offers]
    """
    JT_offers = pd.read_csv('data/matched_esco_jt_offers.csv')
    JT_offers['ESCO'] = ast_string_to_list(JT_offers['ESCO'])
    JT_offers.loc[:,'description_proc'] = JT_offers['description_proc'].astype(str)
    return JT_offers

def get_skills_job_esco(job_title_fr):
    """List of ESCO skills for a ESCO job
    
    Arguments:
        job_title_fr {[str]} -- [ESCO job in french]
    
    Returns:
        [Dictionary] -- [keys are skills and values are (essential or optional)]
    """
    ### this function returns a dictionary of ESCO skills for a specified ESCO Job
    ### Skills DataSet : ESCO skills
    df_skills = retrieve_final_skills()
    ### OccupationSkillRelations DataSet
    df_relation = pd.read_csv("data/occupationSkillRelations.csv")
    ### Occupation fr or eng
    df_occ_fr = pd.read_csv("data/27_lang_ESCO_occupations.csv")

    ## 1 - fetch job
    job = df_occ_fr[df_occ_fr['preferredLabel_fr'] == job_title_fr ]
    ## 2 - fetch url
    url = job.conceptUri.values[0]
    ## 3 - fetch skill links
    Skills_for_job = df_relation[df_relation['occupationUri'] == url][['skillUri','relationType']]
    Skills_for_job = pd.Series(Skills_for_job.relationType.values,index=Skills_for_job.skillUri.values).to_dict()
    
    ## 4 - fetch skills
    Skills_fil = df_skills[df_skills.conceptUri.isin(Skills_for_job.keys())]
    Skills_fil['relationType'] = Skills_fil.loc[:,'conceptUri'].apply(lambda x:Skills_for_job[x])
    Skills_fil = pd.Series(Skills_fil.relationType.values,index=Skills_fil.preferredLabel_y.values).to_dict()

    return Skills_fil

def get_desc_esco(title,job):
    """This function is used in apply() for the Dataframe of JT jobs
    
    Arguments:
        title {[str]} -- [JT offer title]
        job {[str]} -- [ESCO job]
    
    Returns:
        [Boolean] -- [To get the description of a specific ESCO job ]
    """
    for i in title:
        if i == job :
            return True
        else:
            return False

def matching_with_apply(p,x,desc,Dict_Matched):
    """Matching function between Esco alternative labels skills and Jt offers description
    
    Arguments:
        p {[str]} -- [preferredLabel]
        x {[type]} -- [list of alternative labels]
        desc {[type]} -- [Offer description]
        Dict_Matched {[type]} -- [The result of matching: keys : skills preferredLabel,values: frequency]
    """
    for i in x:
        if i in desc:
            if p not in Dict_Matched:
                Dict_Matched[p] = desc.count(i)
            else:
                Dict_Matched[p] +=desc.count(i)

def matching_with_apply_tech(p,x,desc,Dict_Matched,tech):
    """Matching function between Esco alternative labels skills and Jt offers description
    
    Arguments:
        p {[str]} -- [preferredLabel]
        x {[type]} -- [list of alternative labels]
        desc {[type]} -- [Offer description]
        Dict_Matched {[type]} -- [The result of matching: keys : skills preferredLabel, values: frequency]
        tech {[Boolean]} -- [True : tech job, False : non tech job]
    """
    if tech == False:
        if len(x)>1:
            for i in x:
                if i in desc:
                    if p not in Dict_Matched:
                        Dict_Matched[p] = desc.count(i)
                    else:
                        Dict_Matched[p] +=desc.count(i)
    else:
        if len(x) ==1:
            for i in x:
                if i in desc:
                    if p not in Dict_Matched:
                        Dict_Matched[p] = desc.count(i)
                    else:
                        Dict_Matched[p] +=desc.count(i)

def add_space(x):
    """Add space to string
    
    Arguments:
        x {[str]} -- [string]
    
    Returns:
        [str] -- [string with spaces from two sides]
    """
    x = " "+str(x)+" "
    return x

def nettoyage_skills(x):
    """returns list of skills with more than 2 words
    
    Arguments:
        x {[list]} -- [list of skills]
    
    Returns:
        [list] -- [list of skills]
    """
    if len(x)>1 :
        lis = []
        for s in x:
            if len(s.split(" "))>2:
                lis.append(s)
        return lis
    else:
        return x


#### Reduction functions
def function_amod_skill(dep, text, pos):
    """Adjectival modifier reduction method
    
    Arguments:
        dep {[list]} -- [list of dependencies]
        text {[list]} -- [list of words composing the skill]
        pos {[list]} -- [list of positions]
    
    Returns:
        [str] -- [The skill after reduction]
    """
    index = 0

    ind = 99
    if "PUNCT" in pos:
        ind = pos.index("PUNCT")
        aux = text[ind-1:ind+2]
        dep_a = dep[ind+1]
        pos_a = pos[ind+1]
        text[ind-1] = "".join(aux)
        del text[ind:ind+2]
        del dep[ind:ind+2]
        del pos[ind:ind+2]
        dep[ind-1] = dep_a
        pos[ind-1] = pos_a
    Reduced = "No change"
    if "amod" in dep:
        index = dep.index("amod")

        if len(text[index:]) == len(text):
            if "NOUN" in pos[index+1:]:
                index_noun = pos[index+1:].index("NOUN") + 1

                Reduced = " ".join(text[index:index_noun+1])
        else:
            Reduced =  " ".join(text[index:])
        
    if Reduced != " ".join(text):
        return Reduced  + "_amod"
    else:
        return "No change"
    
def function_compound_extraction(dep, text, pos):
    """    # This function extracts compound nouns
    # Compound nouns usually explain the meaning of the skill without the need of verbs or other adjective
    # Example  : manage the operation of propulsion plant machinery -- > propulsion plant machinery
    
    Arguments:
        dep {[list]} -- [list of dependencies]
        text {[list]} -- [list of words composing the skill]
        pos {[list]} -- [list of positions]
    
    Returns:
        [str] -- [The skill after reduction]
    """



    ind = 99
    if "PUNCT" in pos:
        ind = pos.index("PUNCT")
        aux = text[ind-1:ind+2]
        dep_a = dep[ind+1]
        pos_a = pos[ind+1]
        text[ind-1] = "".join(aux)
        del text[ind:ind+2]
        del dep[ind:ind+2]
        del pos[ind:ind+2]
        dep[ind-1] = dep_a
        pos[ind-1] = pos_a

    Reduced = "No change"
    indexs = []
    if "compound" in dep:
        for token in range(len(dep)):
            if dep[token] == "compound":
                indexs.append(token)
        if max(indexs) - min(indexs) == len(indexs) - 1:
            indexs.append(max(indexs)+1)

        else:
            for i in range(len(indexs)-1):
                if indexs[i+1] - indexs[i] != 1:
                    indexs = indexs[:i+1]
                    break
            indexs.append(max(indexs)+1)
        # Get New text and pos
        New_text = list(itemgetter(*indexs)(text))
        # indexs projection on the original text
        # example input: [2,3] ,text = [C1,C2,C3,C4]
        ## output : [C3,C4]
        Reduced = " ".join(New_text)
    else:
        for i in range(len(text)):
            if "-" in text[i]:
                Reduced = text[i]
                break
    if Reduced != " ".join(text):
        return Reduced + "_com"
    else:
        return "No change"


def function_obj_extraction(dep, text, pos):
    """ # This function extracts the first root word and its direct object
    # in a skill phrase, after that we take the substring starting from the ROOT to DOBJ
    # This way we delete every specificty in the skill, such as domaine, field or any entity that could be
    # affected by the skill : perform street interventions in social work -- > perform street interventions
    
    Arguments:
        dep {[list]} -- [list of dependencies]
        text {[list]} -- [list of words composing the skill]
        pos {[list]} -- [list of positions]
    
    Returns:
        [str] -- [The skill after reduction]
    """
    ROOT = 0
    dobj = 0
    for token in range(len(dep)):
        if dep[token] == "ROOT":
            ROOT = token
        if dep[token] == "dobj":
            dobj = token
    Reduced = "No change"
    if (ROOT < dobj):
        Reduced = " ".join(text[ROOT:dobj+1])
    if Reduced == " ".join(text):
        return "No change"
    else:
        return Reduced + "_obj"


def function_compound_dobj_extraction(x):
    """This funciton implements all three methods of reduction and starts with compound nouns, if not found passes to direct object and finaly to Adjectival modifier
    
    Arguments:
        x {[str]} -- [The inital skills]
    
    Returns:
        [str] -- [The skill after reduction]
    """
    Reduced_skill = "No change"
    doc = nlp(x)

    dep = [token.dep_ for token in doc]
    text = [token.text for token in doc]
    pos = [token.pos_ for token in doc]
    # Remove punct from the end if it exists
    if dep[-1] == "punct":
        del dep[-1]
        del text[-1]
        del pos[-1]
    # First case if we find compound nouns in the skill phrase
    if "compound" in dep:
        Reduced_skill = function_compound_extraction(dep, text, pos)
    # Second case if we don't have compound nouns in the skill phrase, but direct obj
    if ("dobj" in dep) and (Reduced_skill == "No change"):
        Reduced_skill = function_obj_extraction(dep, text, pos)
    # Third case
    if ("amod" in dep) and (Reduced_skill == "No change"):
        Reduced_skill = function_amod_skill(dep, text, pos)

    return Reduced_skill

def function_separate_ways(x):
    """This funciton implements all three methods of reduction but returns all them
        if they exist 
    Arguments:
        x {[str]} -- [The inital skills]
    
    Returns:
        [str] -- [The skill after reduction]
    """
    doc = nlp(x)

    dep = [token.dep_ for token in doc]
    text = [token.text for token in doc]
    pos = [token.pos_ for token in doc]
    
    Reduced_skill1 = "No change"
    Reduced_skill2 = "No change"
    Reduced_skill3 = "No change"
    # Remove punct from the end if it exists
    if dep[-1] == "punct":
        del dep[-1]
        del text[-1]
        del pos[-1]
        
    ### First case if we find compound nouns in the skill phrase
    if "compound" in dep:
        Reduced_skill1 = function_compound_extraction(dep,text,pos)
    ### Second case if we don't have compound nouns in the skill phrase, but direct obj
    if "dobj" in dep:
        Reduced_skill2 = function_obj_extraction(dep,text,pos)
    ### Third case
    if "amod" in dep:
        Reduced_skill3 = function_amod_skill(dep,text,pos)
        
    return [Reduced_skill1,Reduced_skill2,Reduced_skill3]