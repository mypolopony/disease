# Wikipedia Disease Classification  
<b>Title</b>          : disease_classifier.py  
<b>Description</b>   : Implementation of text classifiers  
<b>Author</b>         : Selwyn-Lloyd McPherson (selwyn.mcpherson@gmail.com)  
<b>Python Version</b> : 3.5.0

## Contents
0. [Executive Summary](#executive_summary)
1. [Approach and Motivation](#approach_and_motivation)
2. [Implementation](#implementation)
3. [Feature Selection](#feature_selection)
4. [Dataset Size and Performance](#dataset_size_and_performance)
5. [Accuracy](#accuracy)
6. [Decision Function](#decision_function)
7. [Signs and Symptoms](#signs_and_symptoms)
8. [A Note on APIs](#a_note_on_apis)
9. [Identity Resolution](#identity_resolution)
10. [Usage](#usage)
11. [Improvements](#improvements)

<a id='executive_summary'></a>
## Executive Summary
- A Baysean document classifier with 99.3% accuracy
- A context-specific classifier to identify document subsections
- Some clever edge-case considerations
- Many thinkings and considerations (this Readme file)

<a id='approach_and_motivation'></a>
## Approach and Motivation
When it comes to learning the machines, as with most tasks, the first goal is to try to avoid reinventing the wheel. This is an essential and a non-trivial part of how to start thinking about how to solve a problem -- what level of abstraction do we need? What processes do we want under our control, and why?

This comes up all the time when deciding on which libraries to use or which outside services to employ. To any degree that's reasonable, I support dogfooding (https://en.wikipedia.org/wiki/Eating_your_own_dog_food), but where exactly to draw the line is not obvious.

For classification tasks in particular, I wonder if we aren't seeing a real change in the kinds of processes that we are willing to unload onto a third-party. Such is the allure of (X)aaS. And, honestly, we benefit from it all the time.

For the most part, my background is in academic computing and, generally, we ended up cobbling together the software we needed ourselves, usually in relative obscurity. Now there are prediction services galore, and some from reputable organizations (see: Google, IBM). One wonders, how many of the things that we do now will we still be doing in-house a year, or five years, from now?

Anyway. Never be afraid to get your hands dirty!

<a id='implementation'></a>
## Implementation
In this example, I rely on scikit-learn, which is pretty standard. Some people like to go with NLTK, especially with, natural language, but it's not really necessary if one isn't doing lexical analysis. scikit implicitly uses numpy and scipy. I groan slightly at having to drag these hunks of code into a virtual environment, but that's unavoidable. The dependencies also require the LAPACK and BLAS libraries, which are finely tuned packages for for linear algebra and other mathy stuff. They're old and cumbersome and written in FORTRAN, and on a normal system they don't pose any problems, but I must have compiled and installed these two little rascals twenty-something times in the past fifteen years, and it's not a picnic on scientific machines.

<a id='feature_selection'></a>
## Feature Selection
To categorize our articles, we first want to eliminate all of the extraneous HTML and structural elements of the page. First, we pull out anything between the `<p>` tags, clean them up a bit (stemming, formatting, removing non-words, removing stop words), and use the result as our input. I haven't really loved any of the common stemmers available, so I have an alternative that I've used in the past and have had good success with (based on `e-lemma.txt` with explanation in the comments). We remove numbers, links, two-letter words (not helpful), etc. The result is a nice chunk of sanitized words based on the paragraphs (meat and potatoes) of the article.

I use the CountVectorizer and the RandomForestClassifier because I like counting and forests. Actually, it's a slightly arbitrary choice for the purposes of illustration. Naturally, we would test to determine the most high performance combination of vectorizer and classifier, or, if we're feeling good about ourselves, design our own that is tuned to the use case.

The code exists here, as well, to use different parts of the article for training the classifier. For example, we can pick the article's summary, i.e. the first section (if there are sections) or the first paragraph (if there are not). The motivation here is to exploit features of the article that would be of most value. Likewise, there is also code to extract the sidebar, if it exists, as many disease-related articles will have a similar table of contents.

<a id='dataset_size_and_performance'></a>
## Dataset Size and Performance
This dataset, in text form, totals around 1GB. That's nearing the limit of what I'm comfortable with having in memory, and it's not scalable, but we can swing it for now. The CPU load isn't too much for my MacBook Air to handle. Still, training and testing takes about 25 minutes, which is a lot. With proper cloud hardware, that would change.

A quick profile of the training phase reveals the bottlenecks of this particular implementation and can be found in `profiles/training_profile.txt`. This is data from just one run, but it seems to suggest that parsing HTML is the weakest link. I originally hypothesized that I/O would be the limiting factor but it seems that the parsing functions are bearing the brunt of the work. With larger datasets, this can be remedied by alternatives (Spark, Hadoop) to map-reduce the individual files, so it's worthy of some investigation. But for this simple example, I went with pure python mostly because I'm most familiar with it and it is plenty equipped to handle the job.

The implementation here is modular in that it uses pickled binary files to save states, including vectorizers and classifiers and the relevant results. This is a great way to run various iterations of different algorithms and keep track of progress. The structure is labored at times but in the grand scheme of things, I hope here just to isolate the individual pieces and place less emphasis on passing resources around.

In a production environment, we would have a simple database to manage versioning and coordinate data storage.

<a id='accuracy'></a>
## Accuracy
A training set of 60% was chosen as per previous best practices. The ratios of negative to positive were retained.

Initial training and testing results can be found in the pickled files `data/paragraphs.training` and `data/paragraphs.results`

After training, the classifier was used on 30 sets of 1500 random articles to test for accuracy and precision. The average confusion matrix was calculated:

```
INFO Final Confusion Matrix:
[[747   3]
 [  7 743]]
```

With only 7 false positives and 3 false negatives per 1500, the reslts indicates the classifier has an <font color='#0080FF'>**accuracy of 99.3%**</font> (recall = **99.1%**, precision = **99.6%**). Not too bad!

A new profile based on these predictions can be found in `profiles/prediction_profile.txt` reveals that now, our bottleneck seems to come from the stemmer. Humph! That's my function! We'll have to look into a quicker implementation for larger datasets!

<a id='decision_function'></a>
## Decision Function
I hope it's obvious that I think there are many ways to solve any one problem. To the degree that we can identify the best, we should use the best, certainly. But, at least naively, we can define a simple scoring function:

`D = wa*A + wb*B + . . . wn*N`

where, as usual, we have a linear function of the various kinds of analyses we can perform weighted by how certain we are about their particular significance.

The implication is that, for example, we can run multiple classifiers, each with different advantages and disdvantages:

- scikit-learn Count Vectorier + Random Forest (implemented in this example), as well as other classifiers and vectorizers and their parameters
- direct text investigation (implemented in this example, i.e. if first paragraph or sentence contains the form '* is a * disease')
- negative predictors (implemented in this example, i.e. if first paragraph or sentence is of the form '* is a group of diseases' among others) to catch overly-broad articles
- category as indicated by html / DBPedia (see more below)

There are tons of other metrics and outside APIs, though the question quickly becomes existential: what is a disease?

Anyway, the point to stress here is that when we want to classify something, we shouldn't ask whether it is a disease or not, but rather what is the probability, and how confident are we? In this way, we retain clarity and quality assurance as we move forward with more complicated problems.

In essence, you can obtain a matrix like the following:

```
File											RandomForest	has_disease	is_group	dbpedia
articles/positive/Atrophy						1	False	False	False
articles/negative/3065							1	False	False	False
articles/negative/3939							1	False	False	False
articles/positive/Aortic_valve_stenosis			1	False	False	True
articles/positive/Cooks_syndrome				1	True	False	True
articles/positive/Cancer	1	True	True	True
articles/positive/Chondrodysplasia_punctata	1	True	True	False
articles/positive/Watermelon_stomach	1	False	False	False
articles/positive/Dilated_cardiomyopathy	1	False	False	False
articles/negative/9819	1	False	False	False
articles/negative/8644	1	False	False	False
articles/positive/Jumping_Frenchmen_of_Maine	1	False	True	false
```

A random selection of negatives and positives and the various inquries (a **1** in the RandomForest column means a correct guess). This illustrates a few nuances: for example, **Watermelon Stomach** is a colloquial term for **Gastric Antral Vascular Ectasia**. 

**Jumping Frenchmen of Maine**, by the way, is also not a disease per se, but actually a rare and peculiar neuropsychiatric phenomenon observed in lumber workers in the late 1800s. Men in the community became hypersensitive and were easily startled both physically and emotionally, developing sudden tics similar to those found in patients with Tourette Syndrome. Odd!

<a id='signs_and_symptoms'></a>
## Signs and Symptoms
We're also asked to consider a similar categorization task wherein we would like to identify particular aspects of a disease, for example signs and symptoms. Given text, can we pull out sentences or phrases that are related to signs and symptoms?

One simple approach is to make a frequency counter and, after looking at the results and identifying high-frequency words, make a reasonable judgement as to a cut-off, or include this as one prong of a multifactor deision tree.

A more interesting question is to try and identify which words are used to describe signs and symptoms **within** a disease article. That is, are there words that are super-specific for signs and symptoms, even in the midst of a discussion about diseases?

There is a balancing act: for example, in the naive form (i.e. on a general corpus), a non-disease article that happens to mention **pneumonia** and **cough** might get high marks, which would be incorrect. On the other hand, given a set of medical documents, **pneumonia** and **cough** may show up as many times as signs and symptoms as they do outside of that section, which is tricky. 

There's a lot to be said for diving deeper into which words are most valuable (that is, the ones that explain most of the variance), but absent that, it's time to make another classifier.

In this case, I like TF-IDF, especially on a binary comparison, because the cream usually rises to the top. Actually, it log-rises to the top!

Here, we take all of the disease articles, excise the Signs and Symptoms block from the main text (if it exists), and separate the two into bins: section and non-section. Then, using our classifier to train (60%) and test (40%), we get <font color='#0080FF'>**77.0% accuracy**</font>. That's not awful, but it's not as good as our other predictor. I suspect that the space is much more crowded, and the overlap between word-bags is likely to be significant. 

I decided to use bigrams to mitigate the word-overlap, and the phrases that identify the most variance make reasonable sense:

```
red blood			citation need
chest pain 			muscle weakness
patient may 		lymph node
nausea vomit 		may include
may cause 			symptom may
may occur 			blood pressure
abdominal pain 		shortness breath
may present 		sign symptom
symptom include 	weight loss
blood cell 			may also
```

Some of those "may *" bigrams can be safely ignored, clearly, but the others seem to be on point. With this classifier, we also get probabilities and as it turns out, the support for non-matches are indeed weaker than matches (`profiles/sample_run.txt`) 

<a id='a_note_on_apis'></a>
## A Note on APIs
Because this project is not about utilizing external disease databses, this particular discussion is outside the scope of this project, but still fun to think about!
has digested Wikipedia in its entirety and have transformed it into a formal ontology. It is actually a small part of a greater semantic web initiative that, though it never skyrocketed, is still supported by a small number of people and academic groups, mostly in Europe, and is actually quite useful.

SPARQL is the language to talk to these services and query the RDF data. This is accessible publically but for production's sake, should be maintained locally.

The query:
```
PREFIX dbo: <http://dbpedia.org/ontology/>
SELECT * WHERE {
    ?s a dbo:Disease .
}
```

at endpoint: http://dbpedia.org/sparql yields 5425 results.

The values returned from this query are not the same as those given in the problem as positively diseases. But here, things become slightly philosophical. There are lots of articles in the 'positives' folder that are, in fact, not really diseases. **Bronchogenic cyst**, for example, is included in the list of positives, but not in the DBPedia database. In all honesty, **bronchogenic cyst** is not a disease in the formal sense but rather a sequela of a few different disorders.

Likewise, DBPedia cannot be used as a gold standard either. **Colic**, also commonly **baby colic**, is considered by some to be a disease, but it is actually a constellation of symptoms for which there may or may not be an active desease process going on in the background (you have to ask the baby, I guess).

DBPedia has some really fun gems as well: it classifies **shark attack** as a disease, although nowhere does the word **disease** even appear on the Wikipedia page.

To be fair, there exist (in real life!) at least nine ICD-10 codes used in medical billing involving sharks:
```
ICD-10 CODE        DESCRIPTION
----            ----
W56.41XA        Bitten by shark, initial encounter
W56.41XD        Bitten by shark, subsequent encounter
W56.41XS        Bitten by shark, sequela
W56.42XA        Struck by shark, initial encounter
W56.42XD        Struck by shark, subsequent encounter
W56.42XS        Struck by shark, sequela
W56.49XA        Other contact with shark, initial encounter
W56.49XD        Other contact with shark, subsequent encounter
W56.49XS        Other contact with shark, sequela
```

Ontology nightmare.

As an aside, the ICD-10 has plenty of other great catastrophies, like **Struck By Turtle (W59.22XA)** and the very forward-thinking **Spacecraft collision injuring occupant (V95.43XS)**. I'm sure it's gotta happen sometime.

This all begs the question: what we are actually trying to categorize? And most importantly, why?

<a id='identity_resolution'></a>
## Identity Resolution

No discussion on data science would be complete without at least a nod to identity resolution. This is a fun and tricky problem and is essentially the fundamental goal of the work here. Can we build operable ground truths from messy data? 

There are lots of gotchas: baby colic vs. colic. Differences in US / International designations. The disease / symptom / disorder / abnormality / defect plurality. Spelling differences. Eponymous diseases and the (very common) differences in attribution (Stokes–Adams syndrome == Adams–Stokes syndrome == Gerbezius-Morgagni-Adams–Stokes syndrome == Gerbec-Morgagni-Adams–Stokes syndrome). Accents on names, spaces, hyphens. . . 

Building classifiers is relatively easy but building a good classifier on a particular case requires diving into all of these issues, and more.

<a id='usage'></a>
## Usage
After building the virtual environment and installing prerequisites, just run classify.py -- In the it will load the training and testing sets, run some benchmarks, and work on identifying sections. If you're pressed for time, you can find the log of one run at `profiles/sample_run.txt`

<a id='improvements'></a>
## Improvements

- Because this is just a small demonstration, there are a few stylistic changes I would make before the code is production-ready. The first is to settle on a robust, quick pipeline for entry and analysis. Since we have identified some of the major bottlenecks, we can attack each one step-by-step.

- The Article class included here does a fine job, but it could be a little more streamlined. There is some repetition in reading and storing data, which should be mended by thinking more carefully about how and where data lives, both ephemerally and on disk.

- My functions tend to get pretty long (mostly because I come from a scripting background and not a software background) and usually I would refactor every so often, but I think a little bit more can go a long way as far as the current implementation stands. 

- Coded here but not fully implemented is the identification and extraction of features (other than just simply taking the cleaned paragraphs) that might be great at fine-tuning questionable guesses.

- The return types for prediction results are not in an ideal format. Here, I decided to work around it in some unfortunate ways, but for future versions, I would prefer to spec these out a little better. This mostly relates to maintaining associations in unordered data structures but is trivial with some forethought.

- Understand which features are important. Try other classifiers, like Tf-Idf for this particular task.
