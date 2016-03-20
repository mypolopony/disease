#!/usr/bin/env python
#
# title          : disease_classifier.py
# description    : Implementation of text classifiers
# author         : Selwyn-Lloyd McPherson (selwyn.mcpherson@gmail.com)
# python_version : 3.5.0
# ==============================================================================

import re
import glob
import pickle
import logging
import time as time
import YasumasaStemmer as ys

from random import shuffle
from numpy import zeros, int, argsort
from bs4 import BeautifulSoup, element
from SPARQLWrapper import SPARQLWrapper, JSON
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


# Stop words
with open('stop_words.txt','r') as sw:
    stopwords = sw.read().splitlines()
# Logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


class Article():
    '''
    This is a general object for Articles. It handles article-specific functions like pullng out
    sections, cleaning text, and identifying useful elements on the page
    '''

    def __init__(self,filename):
        '''
        Creation of an Article reads in a file and creates a soup. The libxml and html
        packages can be useful and might be quicker, but bs4 higher-level
        '''

        self.filename = filename
        with open (self.filename,'r') as rawfile:
            raw = rawfile.read()
        self.soup = BeautifulSoup(raw, 'html.parser')


    def clean_string(self,text):
        '''
        A utility function for reformatting a text block
        '''

        # Remove non-letters
        review_text = re.sub("[^a-zA-Z]"," ", text)

        # Convert words to lower case and split them
        words = review_text.lower().split()

        # Remove stop words
        words = [w for w in words if not w in stopwords]

        # Remove words of length <= 2 (generally not helpful)
        words = [w for w in words if len(w) > 2]

        # Stem the words
        stemmer = ys.YasumasaStemmer()
        words = [stemmer.stem(w) for w in words]

        return ' '.join(words)


    def get_paragraphs(self):
        '''
        This is the most basic form of extraction: first obtaining a list bs4.element.ResultSet, pulling
        out the text, and finally joining the list to produce a str
        '''

        dirtystring = ' '.join([p.text for p in self.soup.find_all('p')])
        cleanstring = self.clean_string(dirtystring)
        return cleanstring


    def get_summary(self):
        '''
        Articles with an intro section will have a None <p> element, so if we can use that, otherwise
        we'll just use the first paragraph as the summary
        '''

        ps = self.soup.find_all('p')
        if None in ps:
            none_index = ps.index(None)
            return ' '.join([p.text for p in ps[0:none_index]])
        else:
            return ' '.join([p.text for p in ps])


    def get_infobox(self):
        '''
        The infobox table, if it exists, will have table class 'infobox'
        '''

        table = self.soup.find('table', {'class': 'infobox'})
        if table:
            return table.text
        else:
            return None

    def get_sidebar(self):
        '''
        The table of contents, if it exists, will have div class 'toc'
        '''
        toc = self.soup.find('div', {'class':'toc'})
        if toc:
            return toc.text
        else:
            return None

    def get_categories(self,search=None):
        '''
        If we get the text within the scripts tags, which are standard across Wikipedia,
        we can take a sneak look at the categories. Important to not let this fool you into
        complacency! Wikipedia's definition of a disease may not be the same as ours.
        '''
        pattern = re.compile('wgCategories":([^\]]+])')

        cats = self.soup.find_all('script')[1].text
        cats = re.search(pattern,cats)
        if cats.group():
            catstring = cats.group().replace('wgCategories":','')
            if catstring and search:
                if search in catstring:
                    return True
                else:
                    return False
            elif catstring:
                return eval(catstring)
            else:
                return None

    def get_section(self,id='Signs_and_symptoms'):
        '''
        This is a very simple example of extracting a particular section from an article.
        The important thing to keep in mind is that the structure of the sections are as follows:
            <h2>
                <span id='...'></span>
            </h2>
            <p>
            <p>
        Which is kind of odd. But it just means we have to find the proper span, get the parent, and then
        get the next few children that are not h2's.
        '''

        if self.soup.find('span', {'id': id}):
            p_gen = self.soup.find('span', {'id': id}).findParent().nextSiblingGenerator()
            whole = str()
            part = next(p_gen)
            while part.name != 'h2':
                if isinstance(part, element.Tag):
                    whole = whole + self.clean_string(part.text)
                part = next(p_gen)

            return whole
        else:
            return None

    def is_group(self):
        '''
        This is awfully silly: here, we simply look for 'group' in the first sentence.
        '''
        summary = self.get_summary().split('.')
        if 'group of' in summary[0]:
            return True
        else:
            return False

    def has_disease_text(self):
        '''
        A simple text search for disease, disorder, or syndrome in the first line of the article
        '''
        summary = self.get_summary().split('.')
        pattern = re.compile("is a[\w \-,']+(disease|disorder|syndrome)")

        if re.search(pattern,summary[0]):
            return True
        else:
            return False

def train():
    '''
    Main training routine. If the training file does not exist, we'll train a brand new classifier:
    '''
    try:
        training = open('data/paragraphs.training','rb')

        logging.info('Training file read!')
        return training
    except:
        logging.info('Training. . . ')
        st = time.time()

        train_pct = 0.60    # This is chosen ad hoc since there are many thoughts on what
                            # an optimal value should be. Ideally, we would tweak this and
                            # other parameters to find ideal, case-specific values that
                            # yield the best results

        # Get some files!
        negatives = glob.glob('articles/negative/*')
        positives = glob.glob('articles/positive/*')
        shuffle(negatives)      # Shuffling is good for the soul
        shuffle(positives)      # (it also reduces possible correlation)

        # The initialization of this vectorizer is slightly redundant since we handle stopwords elsewhere,
        # but is left in here for illustration
        vectorizer = CountVectorizer(analyzer = "word",
                                     tokenizer = None,
                                     preprocessor = None,
                                     stop_words = stopwords,
                                     max_features = 5000)

        training_set = list()
        training_set_filenames = list()

        # Load and parse negatives
        for n in range(0, int(train_pct*len(negatives))):
            if n % 100 == 0:
                logging.info('Training: Read {} negatives'.format(n))
            article = Article(negatives[n])
            training_set.append((article.get_paragraphs(),'negative'))
            training_set_filenames.append(article.filename)

        # Load and parse positives
        for p in range(0, int(train_pct*len(positives))):
            if p % 100 == 0:
                logging.info('Training: Read {} positives'.format(p))
            article = Article(positives[p])
            training_set.append((article.get_paragraphs(),'positive'))
            training_set_filenames.append(article.filename)

        shuffle(training_set)   # Keep shuffling for entropy!

        train_data_features = vectorizer.fit_transform([t[0] for t in training_set])    # Learn the vocab
        train_data_features = train_data_features.toarray()                             # Feature extraction

        # "Votes are like trees, if you are trying to build a forest. If you have more trees
        # than you have forests, then at that point the pollsters will probably say you will win." (Dan Quayle)
        forest = RandomForestClassifier(n_estimators = 100)
        forest = forest.fit(train_data_features,[t[1] for t in training_set])

        pickle.dump({'vectorizer':vectorizer,
                    'forest':forest,
                     'training_set_filenames':training_set_filenames},
                    open('data/paragraphs.training', 'wb'))

        elapsed_time = time.time() - st
        logging.info('Training saved to paragraphs.training')
        logging.info('Elapsed time: %.2fs' % elapsed_time)

def test():
    '''
    Now that we've trained a model, it's time to test it and see how well we perform!
    :return:
    '''
    try:
        results = open('data/paragraphs.results','rb')
        logging.info('Results file read!')

        return results
    except:
        logging.info('Testing. . .')

        try:
            training = pickle.load(open('data/paragraphs.training','rb'))
        except:
            logging.error('Could not load training file!')

        st = time.time()

        # The following is not the prettiest, but. . . A set difference might be faster, not quite sure
        negatives = list(set(glob.glob('articles/negative/*'))- set(training['training_set_filenames']))
        positives = list(set(glob.glob('articles/positive/*')) - set(training['training_set_filenames']))

        test_set = list()

        for idx,n in enumerate(negatives):
            if idx % 100 == 0:
                logging.info('Testing: Read {} negatives'.format(idx))
            article = Article(n)
            test_set.append((article.filename,article.get_paragraphs()))

        for idx,p in enumerate(positives):
            if idx % 100 == 0:
                logging.info('Testing: Read {} positives'.format(idx))
            article = Article(p)
            test_set.append((article.filename,article.get_paragraphs()))

        shuffle(test_set)

        # A philosophical dilemma here. . . Does the filename go as index 0 or index 1?
        # The same problem appears above, where the category goes first or second to the
        # text. This is certainly a case for a dict but they are bulky for such a small
        # project.
        test_data_features = training['vectorizer'].transform([t[1] for t in test_set])
        test_data_features = test_data_features.toarray()


        results = training['forest'].predict(test_data_features)

        pickle.dump({'results':results,'test_set':test_set},open('data/paragraphs.results','wb'))

        elapsed_time = time.time() - st
        logging.info('Results saved to paragraphs.results')
        logging.info('Elapsed time: %.2fs' % elapsed_time)


def test_group(training_filename,num_to_test,explicit_tests=None):
    '''
    This will test a random group of n predictions. Actually, this function handles a little
    bit more than it should. In a real world example, if we know we are going to be iteratively
    calling many groups, we shouldn't have to glob-glob and shuffle every time. We could
    be smarter about this, but it will do for now.

    Also included is the ability to explicitly test various disease, so we make sure
    we add those at the end as well
    '''
    training = pickle.load(open(training_filename,'rb'))

    negatives = glob.glob('articles/negative/*')
    positives = glob.glob('articles/positive/*')
    shuffle(negatives)
    shuffle(positives)

    # Here we make the choice to split the test group into halves; I think this is actually
    # innocuous and has no bearing on the result, but nice practice anyway
    query_set = negatives[0:int(num_to_test/2)] + positives[0:int(num_to_test/2)]

    if explicit_tests:
        query_set = query_set + explicit_tests

    test_set = list()

    # Again, if I/O is a problem, we would ideally have the articles in a database for quick
    # access, instead of having to read in the file every time.
    for qs in query_set:
        article = Article(qs)
        test_set.append((article.filename,article.get_paragraphs()))

    shuffle(test_set)

    # Vocab
    test_data_features = training['vectorizer'].transform([t[1] for t in test_set])
    test_data_features = test_data_features.toarray()

    # Prediction
    results = training['forest'].predict(test_data_features)

    output = {'results':results,'test_set':test_set}

    return output

def benchmark():
    '''
    Let's get a sense of how we're doing on our predictions. We'll run a few sets and see
    what our overall metrics are.
    '''
    st = time.time()

    num_sets = 1
    set_size = 1500

    # Confusion matrix has a standard 2x2 size, so this can be hard coded and zeroed
    conf_mat = zeros((2,2), dtype=int)

    logging.info('Running benchmarks')
    logging.info('{} sets of {}'.format(num_sets,set_size))
    training = pickle.load(open('data/paragraphs.training','rb'))   # Load classifier

    for ns in range(0,num_sets):
        runtime = time.time()

        logging.info('Working set {}/{}'.format(ns+1,num_sets))
        results = test_group(training, set_size)
        targets = list()

        for targ in results['test_set']:        # Canonical assay function
            if 'negative' in targ[0]:           # This is the URL portion, remember
                targets.append('negative')
            else:
                targets.append('positive')

        conf_mat += confusion_matrix(targets,results['results'])

        elapsed_time = time.time() - runtime
        logging.info('Set completed in: %.2fs' % elapsed_time)

    logging.info('Final Confusion Matrix:')
    logging.info(print(confusion_matrix(targets,results['results'])))

    elapsed_time = time.time() - st
    logging.info('Elapsed time: %.2fs' % elapsed_time)

def get_dbpedia_diseases():
    '''
    Simple call to request the list of items labeled as diseases in DBPedia
    '''
    sparql = SPARQLWrapper('http://dbpedia.org/sparql')
    sparql.setQuery("""
        PREFIX dbo: <http://dbpedia.org/ontology/>
        SELECT * WHERE {
        ?s a dbo:Disease .
        }
    """)

    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    dbpedia_diseases = [r['s']['value'] for r in results['results']['bindings']]
    dbpedia_diseases = [d.replace('http://dbpedia.org/resource/','') for d in dbpedia_diseases]

    return dbpedia_diseases

def sanity():
    '''
    Out of curiosity, how many of the 'positive' articles have the word disease in them?
    '''
    logging.info('Performing Sanity Check')

    st = time.time()


    positives = glob.glob('articles/positive/*')
    count = 0
    for p in positives:
        a = Article(p)
        if a.has_disease_text():
            count += 1
    logging.info('Positives that contain \'disease\': {}/{}'.format(count,len(positives)))

    elapsed_time = time.time() - st
    logging.info('Elapsed time: %.2fs' % elapsed_time)


def classify_rf():
    '''
    Here, I choose to isolate the initial training and testing from actual predictive
    functionality. It's a modular style that is just used for illustration. It gets
    a bit awkward and leads to some repetition, but, again, this is a demonstration
    not intended to showcase methodologies, not be as efficient as possible.
    '''

    train()             # Train
    test()              # Test
    benchmark()        # Benchmark
    sanity()           # Are diseases really diseases?


    # So now let's run a handful of random articles through our tests and come up with
    # a solution to the decision functions. Since this is the actual use case, the functions
    # will return quickly if training and testing data has been done previously.
    num_to_test = 30
    logging.info('Analyzing Query with {num} samples'.format(num=num_to_test))

    # Random Forest (with explicit_tests added)
    explicit_tests = ['articles/positive/Cancer','articles/positive/Warfarin_necrosis']
    group_results = test_group('data/paragraphs.training',num_to_test,explicit_tests)

    # Refactoring results (see Improvements Section in Readme)
    results = dict()
    for idx,gr in enumerate(group_results['test_set']):
        filename = gr[0]
        results[filename] = dict()
        results[filename]['rf_prediction'] = group_results['results'][idx]

    for result in results.keys():
        if results[result]['rf_prediction'] in result:  # Match!
            results[result]['random_forest'] = 1
        else:
            results[result]['random_forest'] = 0

    # Is disease? Group? DBPedia?
    dbpedia_diseases = get_dbpedia_diseases()

    for q in results.keys():
        a = Article(q)
        results[q]['has_disease_text'] = a.has_disease_text()
        results[q]['is_group'] = a.is_group()
        results[q]['has_category'] = a.get_categories('disease')
        if q.split('/')[-1] in dbpedia_diseases:
            results[q]['dbpedia'] = True
        else:
            results[q]['dbpedia'] = False

    logging.info('Results. . .')
    logging.info('File\tRandomForest\thas_disease\tis_group\tdbpedia')
    for q in results.keys():
        logging.info('{}\t{}\t{}\t{}\t{}'.format(q,
                                             results[q]['random_forest'],
                                             results[q]['has_disease_text'],
                                             results[q]['is_group'],
                                             results[q]['has_category'],
                                             results[q]['dbpedia']))

    '''
    Note that the prediction will now be something of the following:
    Prediction = w1*(RandomForest) + w2*(is_disease) - w3*(is_group_disease) + w4*(has_category_disease)
    '''

def classify_sections(id='Signs_and_symptoms'):
    '''
    Here, we're going to see if we can train a classifier that can pick out a particular section
    (labeled with id). We'll limit our cases to positive disease articles in the hope tha the
    classiifier might be very specific
    '''

    logging.info('Learning sections: {id}'.format(id=id))

    st = time.time()

    positives = glob.glob('articles/positive/*')
    train_pct = 0.60

    # Develop training and testing lists
    shuffle(positives)
    training = positives[0:int(train_pct*len(positives))]
    testing = list(set(positives)-set(training))

    section_set = list()
    non_section_set = list()

    logging.info('Reading files for training. . .')
    for tfile in training:
        a = Article(tfile)
        paragraphs = a.get_paragraphs()
        section = a.get_section(id)

        # We might not have a section, which is fine
        if section:
            # Now for the niftiest little trick in the book (it really does work!):
            paragraphs = paragraphs.replace(section,'')
            section_set.append(section)

        non_section_set.append(paragraphs)

    logging.info('Training. . .')

    # First we need to get counts of words in our corpus
    count_vect = CountVectorizer(lowercase = True,
                                 stop_words = stopwords,
                                 ngram_range = (2,2))
    train_counts = count_vect.fit_transform(section_set + non_section_set)

    # Now transform to frequency space
    tfidf_transformer = TfidfTransformer()
    train_tfidf = tfidf_transformer.fit_transform(train_counts)

    targets = ['section' for s in section_set] + ['non_section' for ns in non_section_set]
    classifier = MultinomialNB().fit(train_tfidf,targets)

    logging.info('Done training!')

    elapsed_time = time.time() - st
    logging.info('Elapsed time: %.2fs' % elapsed_time)

    # Now test:
    logging.info('Testing sections: {id}'.format(id=id))

    st = time.time()

    section_set = list()
    non_section_set = list()

    logging.info('Reading files for testing')
    # There is some repetition here that would be tightened up in production
    for tfile in testing:
        a = Article(tfile)
        paragraphs = a.get_paragraphs()
        section = a.get_section(id)

        if section:
            paragraphs = paragraphs.replace(section,'')
            section_set.append(section)

        non_section_set.append(paragraphs)

    targets = ['section' for s in section_set] + ['non_section' for ns in non_section_set]

    logging.info('Predicting. . .')
    # Run the testing data through the vocab-grabber
    new_counts = count_vect.transform(section_set + non_section_set)

    # New TF-IDF transform on the testing data
    new_tfidf = tfidf_transformer.transform(new_counts)

    # And predictions!
    predicted = classifier.predict(new_tfidf)

    # Grab the feature names and print the most valuable 20
    feature_names = count_vect.get_feature_names()
    top20 = argsort(classifier.coef_[0])[-20:]
    for idx,t in enumerate(top20):
        print('{} ({})'.format(feature_names[t],idx))

    correct_count = 0

    for sample in range(0,len(predicted)):
        # For reasons, the string representation is best determined out here
        probstr = '{:.2f}'.format(max(classifier.predict_proba(new_tfidf)[sample]))
        print('{}\t{}\t{}'.format(predicted[sample],targets[sample],probstr))

        if predicted[sample] == targets[sample]:
            correct_count += 1

    logging.info('Percentage correct: {:.2f}'.format(correct_count/len(predicted)))

    pickle.dump({'classifier':classifier,
                 'count_vec':count_vect,
                 'new_tfidf':new_tfidf,
                 'new_counts':new_counts,
                 'predicted':predicted,
                 'targets':targets},open('data/section_predicted.dat','wb'))

    logging.info('Done testing!')

    elapsed_time = time.time() - st
    logging.info('Elapsed time: %.2fs' % elapsed_time)

if __name__ == '__main__':
    classify_rf()                           # Classifying documents as disease or non-disease using Random Forests
    classify_sections('Signs_and_symptoms') # Classifying document sections with TF-IDF
