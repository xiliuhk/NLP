Created by @ttezel
https://gist.github.com/ttezel/4138642

#A Collection of NLP notes

##N-grams

###Calculating unigram probabilities:

P( w<sub>i</sub> ) = count ( w<sub>i</sub> ) ) / count ( total number of words )

In english..

Probability of word<sub>i</sub> =
Frequency of word (i) in our corpus / total number of words in our corpus

##Calcuting bigram probabilities:

P( w<sub>i</sub> | w<sub>i-1</sub> ) = count ( w<sub>i-1</sub>, w<sub>i</sub> ) / count ( w<sub>i-1</sub> )

In english..

Probability that word<sub>i-1</sub> is followed by word<sub>i</sub> =
[Num times we saw word<sub>i-1</sub> followed by word<sub>i</sub>] / [Num times we saw word<sub>i-1</sub>]

Example
-------
- **s** = beginning of sentence
- **/s** = end of sentence

####Given the following corpus:

**s** I am Sam **/s**

**s** Sam I am **/s**

**s** I do not like green eggs and ham **/s**

We can calculate bigram probabilities as such:

- P( I | **s** ) = 2/3 

=> Probability that an **s** is followed by an `I`

= [Num times we saw `I` follow **s** ] / [Num times we saw an **s** ]
= 2 / 3

- P( Sam | am ) = 1/2

=> Probability that `am` is followed by `Sam`

= [Num times we saw `Sam` follow `am` ] / [Num times we saw `am`]
= 1 / 2

###Calculating trigram probabilities:

Building off the logic in bigram probabilities,

P( w<sub>i</sub> | w<sub>i-1</sub> w<sub>i-2</sub> ) = count ( w<sub>i</sub>, w<sub>i-1</sub>, w<sub>i-2</sub> ) / count ( w<sub>i-1</sub>, w<sub>i-2</sub> )

In english...

Probability that we saw word<sub>i-1</sub> followed by word<sub>i-2</sub> followed by word<sub>i</sub> = [Num times we saw the three words in order] / [Num times we saw word<sub>i-1</sub> followed by word<sub>i-2</sub>]

Example
-------

- P( Sam | I am ) 
= count( Sam I am ) / count(I am)
= 1 / 2

###Interpolation using N-grams

We can combine knowledge from each of our n-grams by using interpolation.

E.g. assuming we have calculated unigram, bigram, and trigram probabilities, we can do:

P ( Sam | I am ) = &Theta;<sub>1</sub> x P( Sam ) + &Theta;<sub>2</sub> x P( Sam | am ) + &Theta;<sub>3</sub> x P( Sam | I am )

Using our corpus and assuming all lambdas = 1/3,

P ( Sam | I am ) = (1/3)x(2/20) + (1/3)x(1/2) + (1/3)x(1/2)

In web-scale applications, there's too much information to use interpolation effectively, so we use **Stupid Backoff** instead.

In **Stupid Backoff**, we use the trigram if we have enough data points to make it seem credible, otherwise if we don't have enough of a trigram count, we back-off and use the bigram, and if there still isn't enough of a bigram count, we use the unigram probability.

###Smoothing Algorithms

Let's say we've calculated some n-gram probabilities, and now we're analyzing some text. What happens when we encounter a word we haven't seen before? How do we know what probability to assign to it?

We use `smoothing` to give it a probability.

=> Use the count of things we've only seen *once* in our corpus to estimate the count of things we've *never seen*.

This is the intuition used by many smoothing algorithms.

###Good-Turing Smoothing

Notation:

N<sub>c</sub> = the count of things with frequency `c` - how many *things* occur with frequency `c` in our corpus.

Good Turing modifies our:
- n-gram probability function for things we've never seen (things that have count 0)
- count for things we *have* seen (since all probabilites add to 1, we have to modify this count if we are introducing new probabilities for things we've never seen)

Modified Good-Turing probability function:

P<sup>*</sup> ( things with 0 count ) = N<sub>1</sub> / N

=> [Num things with frequency 1] / [Num things]

Modified Good-Turing count:

count<sup>*</sup> = [ (count + 1) x N<sub>c+1</sub> ] / [ N<sub>c</sub> ]

Example
-------

Assuming our corpus has the following frequency count:

carp:       10
perch:      3
whitefish:  2
trout:      1
salmon:     1
eel:        1

Calculating the probability of something we've never seen before:

P ( catfish ) = N<sub>1</sub> / N = 3 / 18

Calculating the modified count of something we've seen:

count<sup>*</sup> ( trout )

= [ (1 + 1) x N<sub>2</sub> ] / [ N<sub>1</sub> ]
= [ 2 x 1 ] / [ 3 ]
= 2 / 3

Calculating the probability of something we've seen:

P<sup>*</sup> ( trout ) = count ( trout ) / count ( all things ) = (2/3) / 18 = 1/27

What happens if we don't have a word that occurred exactly N<sub>c+1</sub> times?

=> Once we have a sufficient amount of training data, we generate a best-fit curve to make sure we can calculate an estimate of N<sub>c+1</sub> for any `c`.

###Kneser-Ney Smoothing

A problem with Good-Turing smoothing is apparent in analyzing the following sentence, to determine what word comes next:

```
I can't see without my reading ___________
```

The word `Francisco` is more common than the word `glasses`, so we may end up choosing `Francisco` here, instead of the correct choice, `glasses`.

The Kneser-Ney smoothing algorithm has a notion of `continuation probability` which helps with these sorts of cases. It also saves you from having to recalculate all your counts using `Good-Turing` smoothing.

Here's how you calculate the K-N probabilty with bigrams:

P<sub>kn</sub>( w<sub>i</sub> | w<sub>i-1</sub> ) = [ max( count( w<sub>i-1</sub>, w<sub>i</sub> ) - `d`, 0) ] / [ count( w<sub>i-1</sub> ) ] + &Theta;( w<sub>i-1</sub> ) x P<sub>continuation</sub>( w<sub>i</sub> )

Where:

**P<sub>continuation</sub>( w<sub>i</sub> )** 

represents the continuation probability of w<sub>i</sub>. This is the number of bigrams where w<sub>i</sub> followed w<sub>i-1</sub>, divided by the total number of bigrams that appear with a frequency > 0. It gives an indication of the probability that a given word will be used as the second word in an unseen bigram (such as `reading ________`)

**&Theta;( )**
This is a `normalizing constant`; since we are subtracting by a `discount weight` **d**, we need to re-add that probability mass we have discounted. It gives us a weighting for our P<sub>continuation</sub>.

We calculate this as follows:

&Theta;( w<sub>i-1</sub> ) = { d * [ Num words that can follow w<sub>i-1</sub> ] } / [ count( w<sub>i-1</sub> ) ]

###Kneser-Ney Smoothing for N-grams

The Kneser-Ney probability we discussed above showed only the bigram case.

For N-grams, the probability can be generalized as follows:

P<sub>kn</sub>( w<sub>i</sub> | w<sub>i-n+1</sub><sup>i-1</sup>) = [ max( count<sub>kn</sub>( w<sub>i-n+1</sub><sup>i</sup> ) - `d`, 0) ] / [ count<sub>kn</sub>( w<sub>i-n+1</sub><sup>i-1</sup> ) ] + &Theta;( w<sub>i-n+1</sub><sup>i-1</sup> ) x P<sub>kn</sub>( w<sub>i</sub> | w<sub>i-n+2</sub><sup>i-1</sup> )

Where:

c<sub>kn</sub>(&bull;) =

- the actual count(&bull;) for the highest order n-gram

or

- continuation_count(&bull;) for lower order n-gram

=> continuation_count = Number of unique single word contexts for &bull;

##Spelling Correction

We can imagine a noisy channel model for this (representing the keyboard).

`original word` ~~~~~~~~~Noisy Channel~~~~~~~~> `noisy word`

Our decoder receives a `noisy word`, and must try to guess what the `original` (intended) word was.

So what we can do is generate **N** possible `original words`, and run them through our **noisy channel** and see which one looks most like the `noisy word` we received.

The corrected word, w<sup>*</sup>, is the word in our vocabulary (`V`) that has the maximum probability of being the correct word (`w`), given the input `x` (the misspelled word).

w<sup>*</sup> = argmax<sub>w&isin;V</sub> P( w | x )

Using Bayes' Rule, we can rewrite this as:

w<sup>*</sup> = argmax<sub>w&isin;V</sub> P( x | w ) x P( w )

**P( x | w )** is determined by our **channel model**.
**P( w )** is determined by our **language model** (using N-grams).

The first thing we have to do is generate candidate words to compare to the misspelled word.

###Confusion Matrix

This is how we model our noisy channel. A confusion matrix gives us the probabilty that a given spelling mistake (or word edit) happened at a given location in the word. We use the **Damerau-Levenshtein** edit types (deletion, insertion, substitution, transposition). These account for 80% of human spelling errors.

- del[a,b]: count( ab typed as a )
- ins[a,b]: count( a typed as ab )
- sub[a,b]: count( a typed as b )
- trans[a,b]: count( ab typed as ba )

Our confusion matrix keeps counts of the frequencies of each of these operations for each letter in our alphabet, and from this matrix we can generate probabilities.

We would need to train our confusion matrix, for example using wikipedia's list of common english word misspellings.

After we've generated our confusion matrix, we can generate probabilities.

Let w<sub>i</sub> denote the i<sup>th</sup> character in the word **w**.

**p( x | w ) =**

- if deletion, [ del( w<sub>i-1</sub>, w<sub>i</sub> ) ] / [ count(w<sub>i-1</sub> w<sub>i</sub>) ]
- if insertion, [ ins( w<sub>i-1</sub>, x<sub>i</sub> ) ] / [ count(w<sub>i-1</sub> ]
- if substitution, [ sub( x<sub>i</sub>, w<sub>i</sub> ) ] / [ count(w<sub>i</sub> ]
- if transposition, [ trans( w<sub>i</sub>, w<sub>i+1</sub> ) ] / [ count(w<sub>i</sub> w<sub>i+1</sub> ]

Suppose we have the misspelled word **x** = **acress**

We can generate our channel model for **acress** as follows:

**actress**

=> Correct letter   : `t`

=> Error letter     : `-`

=> **x | w**        : `c` | `ct` (probability of deleting a `t` given the correct spelling has a `ct`)

=> P( x | w )       : 0.000117

**cress**

=> Correct letter   : `-`

=> Error letter     : `a`

=> **x | w**        : `a` | `-`

=> P( x | w )       : 0.00000144

**caress**

=> Correct letter   : `ca`

=> Error letter     : `ac`

=> **x | w**        : `ac` | `ca`

=> P( x | w )       : 0.00000164

... and so on

We would combine the information from out channel model by multiplying it by our n-gram probability.

###Real-Word Spelling Correction

What happens when a user misspells a word as another, **valid** english word?

Eg. I have fifteen **minuets** to leave the house.

We find valid english words that have an edit distance of 1 from the input word.

Given a sentence w<sub>1</sub>, w<sub>2</sub>, w<sub>3</sub>, ..., w<sub>n</sub>

Generate a set of candidate words for each w<sub>i</sub>

- Candidate( w<sub>1</sub> ) = { w<sub>1</sub>, w<sub>1</sub><sup>'</sup>, w<sub>1</sub><sup>''</sup>, ... }
- Candidate( w<sub>2</sub> ) = { w<sub>2</sub>, w<sub>2</sub><sup>'</sup>, w<sub>2</sub><sup>''</sup>, ... }
- Candidate( w<sub>n</sub> ) = { w<sub>n</sub>, w<sub>n</sub><sup>'</sup>, w<sub>n</sub><sup>''</sup>, ... }

Note that the candidate sets include the original word itself (since it may actually be correct!)

Then we choose the sequence of candidates **W** that has the maximal probability.

Example
-------

Given the sentence `two of thew`, our sequences of candidates may look like:

- two of thew
- two of the
- to off threw
- to off the
- to on threw
- to on the
- to of threw
- to of the
- too of threw
- too of the

Then we ask ourselves, of all possible sentences, which has the highest probability?

In practice, we simplify by looking at the cases where only 1 word of the sentence was mistyped (note that above we were considering all possible cases where each word could have been mistyped). So we look at all possibilities with one word replaced at a time. This changes our run-time from O(n<sup>2</sup>) to O(n).

Where do we get these probabilities?

- Our language model (unigrams, bigrams, ..., n-grams)
- Our Channel model (same as for non-word spelling correction)

Our Noisy Channel model can be further improved by looking at factors like:

- The nearby keys in the keyboard
- Letters or word-parts that are pronounced similarly (such as `ant`->`ent`)

##Text Classification

Text Classification allows us to do things like:

- determining if an email is spam
- determining who is the author of some piece of text
- determining the likelihood that a piece of text was written by a man or a woman
- Perform sentiment analysis on some text

Let's define the Task of Text Classification

Given:

- a document **d**
- a fixed set of classes **C = { c<sub>1</sub>, c<sub>2</sub>, ... , c<sub>n</sub> }**

Determine:

- the predicted class **c &isin; C**

Put simply, we want to take a piece of text, and assign a class to it.

###Classification Methods

We can use **Supervised Machine Learning**:

Given:

- a document **d**
- a fixed set of classes **C = { c<sub>1</sub>, c<sub>2</sub>, ... , c<sub>n</sub> }**
- a training set of **m** documents that we have pre-determined to belong to a specific class

We train our classifier using the training set, and result in a learned classifier.

We can then use this learned classifier to classify new documents.

Notation: we use &Upsilon;(d) = C to represent our classifier, where **&Upsilon;()** is the classifier, **d** is the document, and **c** is the class we assigned to the document.

(Google's `mark as spam` button probably works this way).

####Naive Bayes Classifier

This is a simple (naive) classification method based on Bayes rule. It relies on a very simple representation of the document (called the **bag of words** representation)

Imagine we have 2 classes ( **positive** and **negative** ), and our input is a text representing a review of a movie. We want to know whether the review was **positive** or **negative**. So we may have a bag of positive words (e.g. `love`, `amazing`, `hilarious`, `great`), and a bag of negative words (e.g. `hate`, `terrible`). 

We may then count the number of times each of those words appears in the document, in order to classify the document as **positive** or **negative**.

This technique works well for **topic** classification; say we have a set of academic papers, and we want to classify them into different topics (computer science, biology, mathematics).

####Bayes' Rule applied to Documents and Classes

For a document **d** and a class **c**, and using Bayes' rule,

P( c | d ) = [ P( d | c ) x P( c ) ] / [ P( d ) ]

The class mapping for a given document is the class which has the maximum value of the above probability.

Since all probabilities have P( d ) as their denominator, we can eliminate the denominator, and simply compare the different values of the numerator:

P( c | d ) = P( d | c ) x P( c ) 

Now, what do we mean by the term **P( d | c )** ?

Let's represent the document as a set of features (words or tokens) **x<sub>1</sub>, x<sub>2</sub>, x<sub>3</sub>, ...**

We can then re-write **P( d | c )** as:

P( x<sub>1</sub>, x<sub>2</sub>, x<sub>3</sub>, ... , x<sub>n</sub> | c )

What about P( c ) ? How do we calculate it?

=> P( c ) is the **total probability** of a class. 
=> How often does this class occur in total?

E.g. in the case of classes **positive** and **negative**, we would be calculating the probability that any given review is **positive** or **negative**, without actually analyzing the current input document.

This is calculated by counting the relative frequencies of each class in a corpus.

E.g. out of 10 reviews we have seen, 3 have been classified as **positive**.

=> P ( positive ) = 3 / 10

Now let's go back to the first term in the Naive Bayes equation: 

**P( d | c )**, or **P( x<sub>1</sub>, x<sub>2</sub>, x<sub>3</sub>, ... , x<sub>n</sub> | c )**.

How do we actually calculate this?
----------------------------------

We use some assumptions to simplify the computation of this probability:

- the **Bag of Words assumption** => assume the position of the words in the document doesn't matter.
- **Conditional Independence** => Assume the feature probabilities **P( x<sub>i</sub> | c<sub>j</sub> )** are independent given the class **c**.

It is important to note that both of these assumptions aren't actually correct - of course, the order of words matter, and they are not independent. A phrase like `this movie was incredibly terrible` shows an example of how both of these assumptions don't hold up in regular english. 

However, these assumptions greatly simplify the complexity of calculating the classification probability. And in practice, we can calculate probabilities with a reasonable level of accuracy given these assumptions.

So..
------
To calculate the Naive Bayes probability, **P( d | c ) x P( c )**, we calculate P( x<sub>i</sub> | c ) for each x<sub>i</sub> in **d**, and multiply them together. 

Then we multiply the result by P( c ) for the current class. We do this for each of our classes, and choose the class that has the maximum overall value.

###How do we learn the values of **P ( c )** and **P ( x<sub>i</sub> | c )** ?
------------------------------------------------------------------------------

=> We can use **Maximum Likelihood estimates**.

Simply put, we look at frequency counts.

P ( c<sub>i</sub> ) = [ Num documents that have been classified as c<sub>i</sub> ] / [ Num documents ]

In english..

Out of all the documents, how many of them were in class **i** ?

P ( w<sub>i</sub> | c<sub>j</sub> ) = [ count( w<sub>i</sub>, c<sub>j</sub> ) ] / [ &Sigma;<sub>w&isin;V</sub> count ( w, c<sub>j</sub> ) ]

In english...

The probability of word **i** given class **j** is the count that the word occurred in documents of class **j**, divided by the sum of the counts of each word in our vocabulary in class **j**.

So for the denominator, we iterate thru each word in our vocabulary, look up the frequency that it has occurred in class **j**, and add these up.

####Problems with Maximum-Likelihood Estimate.

What if we haven't seen any training documents with the word **fantastic** in our class **positive** ?

In this case, P ( **fantastic** | **positive** ) = 0

=> This is **BAD**

Since we are calculating the overall probability of the class by multiplying individual probabilities for each word, we would end up with an overall probability of **0** for the **positive** class.

So how do we fix this issue?

We can use a Smoothing Algorithm, for example **Add-one smoothing** (or **Laplace smoothing**).

####Laplace Smoothing

We modify our conditional word probability by adding 1 to the numerator and modifying the denominator as such:

P ( w<sub>i</sub> | c<sub>j</sub> ) = [ count( w<sub>i</sub>, c<sub>j</sub> ) + 1 ] / [ &Sigma;<sub>w&isin;V</sub>( count ( w, c<sub>j</sub> ) + 1 ) ]

This can be simplified to

P ( w<sub>i</sub> | c<sub>j</sub> ) = [ count( w<sub>i</sub>, c<sub>j</sub> ) + 1 ] / [ &Sigma;<sub>w&isin;V</sub>( count ( w, c<sub>j</sub> ) ) + |V| ]

where |V| is our vocabulary size (we can do this since we are adding 1 for each word in the vocabulary in the previous equation).

####So in Summary, to Machine-Learn your Naive-Bayes Classifier:

Given:

- an input document
- the category that this document belongs to

We do:

- increment the count of total documents we have learned from **N**.
- increment the count of documents that have been mapped to this category **N<sub>c</sub>**.
- if we encounter new words in this document, add them to our vocabulary, and update our vocabulary size **|V|**.
- update count( w, c ) => the frequency with which each word in the document has been mapped to this category.
- update count ( c ) => the total count of all words that have been mapped to this class.

So when we are confronted with a new document, we calculate for each class:
---------------------------------------------------------------------------

**P( c )** = N<sub>c</sub> / N

=> how many documents were mapped to class **c**, divided by the total number of documents we have ever looked at. This is the **overall**, or **prior** probability of this class.

Then we iterate thru each word in the document, and calculate:

**P( w | c )** = [ count( w, c ) + 1 ] / [ count( c ) + |V| ]

=> the count of how many times this word has appeared in class **c**, plus 1, divided by the total count of all words that have ever been mapped to class **c**, plus the vocabulary size.
This uses the Laplace-Smoothing, so we don't get tripped up by words we've never seen before. This equation is used both for words we **have** seen, as well as words we **haven't** seen.

=> we multiply each **P( w | c )** for each word **w** in the new document, then multiply by **P( c )**, and the result is the **probability that this document belongs to this class**.

####Some Ways that we can tweak our Naive Bayes Classifier

Depending on the domain we are working with, we can do things like

- Collapse Part Numbers or Chemical Names into a single token
- Upweighting (counting a word as if it occurred twice)
- Feature selection (since not all words in the document are usually important in assigning it a class, we can look for specific words in the document that are good indicators of a particular class, and drop the other words - those that are viewed to be **semantically empty**)

=> If we have a sentence that contains a **title** word, we can upweight the sentence (multiply all the words in it by 2 or 3 for example), or we can upweight the **title** word itself (multiply it by a constant).

##Sentiment Analysis

###Scherer Typology of Affective States

- **Emotion**

Brief, organically synchronized.. evaluation of a major event
=> angry, sad, joyful, fearful, ashamed, proud, elated

- **Mood**

diffuse non-caused low-intensity long-duration change in subjective feeling
=> cheerful, gloomy, irritable, listless, depressed, buoyant

- **Interpersonal Stances**

Affective stance towards another person in a specific interaction
=> friendly, flirtatious, distant, cold, warm, supportive, contemtuous

- **Attitudes**

Enduring, affectively colored beliefs, disposition towards objects or persons
=> liking, loving, hating, valuing, desiring

- **Personality Traits*

Stable personality dispositions and typical behavior tendencies
=> nervous, anxious, reckless, morose, hostile, jealous


**Sentiment Analysis** is the detection of **attitudes** (2nd from the bottom of the above list).

We want to know:
---------------

- The **Holder** (source) of the attitude

- The **Target** (aspect) of the attitude

- The **Type** of the attitude from a set of types (like, love, hate, value, desire, etc.).
Or, more commonly, simply the weighted polarity (positive, negative, neutral, together with **strength**).

###Baseline Algorithm for Sentiment Analysis

Given a piece of text, we perform:

- Tokenization
- Feature Extraction
- Classification using different classifiers
    - Naive Bayes
    - MaxEnt
    - SVM

####Tokenization Issues

Depending on what type of text we're dealing with, we can have the following issues:

- Dealing with HTML or XML markup
- Twitter Markup (names, hash tags)
- Capitalization
- Phone Numbers, dates, emoticons

Some useful code for tokenizing:

- Christopher Potts Sentiment Tokenizer
- Brendan O'Connor Twitter Tokenizer  

####Classification

We will have to deal with handling negation:

`I didn't like this movie` **vs** `I really like this movie`

####So, how do we handle negation?

One way is to prepend **NOT_** to every word between the negation and the beginning of the next punctuation character.

E.g. I didn't really like this movie, but ...

=> I didn't NOT_really NOT_like NOT_this NOT_movie, but ...

This doubles our vocabulary, but helps in tokenizing negative sentiments and classifying them.

####Hatzivassiloglou and McKeown intuition for identifying word polarity

- Adjectives conjoined by **and** have the same polarity

=> Fair **and** legitimate, corrupt **and** brutal

- Adjectives conjoined by **but** do not

=> Fair **but** brutal

We can use this intuition to **learn** new adjectives.

Imagine we have a set of adjectives, and we have identified the polarity of each adjective. Whenever we see a new word we haven't seen before, and it is joined to an adjective we have seen before by an **and**, we can assign it the same polarity.

For example, say we know the poloarity of **nice**.

When we see the phrase `nice and helpful`, we can learn that the word **helpful** has the same polarity as the word **nice**. In this way, we can learn the polarity of new words we haven't encountered before.

So we can expand our **seed set** of adjectives using these rules. Then, as we count the frequency that **but** has occurred between a pair of words versus the frequency with which  **and** has occurred between the pair, we can start to build a ratio of **but**s to **and**s, and thus establish a degree of polarity for a given word.

####What about learning the polarity of phrases?

- Take a corpus, and divide it up into phrases.

Then run through the corpus, and extract the **first two words of every phrase** that matches one these rules:

- 1st word is adjective, 2nd word is noun_singular or noun_plural, 3rd word is **anything**
- 1st word is adverb, 2nd word is adjective, 3rd word is NOT noun_singular or noun_plural
- 1st word is adjective, 2nd word is adjective, 3rd word is NOT noun_singular or noun_plural
- 1st word is noun_singular or noun_plural, 2nd word is adjective, 3rd word is NOT noun_singular or noun_plural
- 1st word is adverb, 2nd word is verb, 3rd word is anything

Note: To do this, we'd have to run each phrase through a Part-of-Speech tagger.

Then, we can look at how often they co-occur with positive words.

- Positive phrases co-occur more with **excellent**
- Negative phrases co-occur more with **poor**

But how do we measure co-occurrence?

We can use Pointwise Mutual Information:

How much more do events **x** and **y** occur than if they were independent?

PMI( word<sub>1</sub>, word<sub>2</sub> ) = log<sub>2</sub> { [ P( word<sub>1</sub>, word<sub>2</sub> ] / [ P( word<sub>1</sub> ) x P( word<sub>2</sub> ) ] }

Then we can determine the polarity of the phrase as follows:

**Polarity( phrase )** = PMI( phrase, **excellent** ) - PMI( phrase, **poor** )

= log<sub>2</sub> { [ P( phrase, **excellent** ] / [ P( phrase ) x P( **excellent** ) ] } - log<sub>2</sub> { [ P( phrase, **poor** ] / [ P( phrase ) x P( **poor** ) ] } 

Another way to learn polarity (of words)
----------------------------------------

Start with a seed set of **positive** and **negative** words.

Then, look them up in a thesaurus, and:

- add synonyms of each of the **positive** words to the **positive** set
- add antonyms of each of the **positive** words to the **negative** set

- add synonyms of each of the **negative** words to the **negative** set
- add antonyms of each of the **negative** words to the **positive** set

and.. repeat.. with the new set of words we have discovered, to build out our lexicon.

###Summary on learning Lexicons

- Start with a *seed set** of words ( `good`, `poor`, ... )
- Find other words that have similar polarity:
    - using **and** and **but**
    - using words that appear nearby in the same document
    - using synonyms and antonyms

###Sentiment Aspect Analysis

What happens if we get the following phrase:

`The food was great, but the service was awful.`

This phrase doesn't really have an overall sentiment; it has two separate sentiments; **great food** and **awful service**. So sometimes, instead of trying to tackle the problem of figuring out the overall sentiment of a phrase, we can instead look at finding the **target** of any sentiment.

How do we do this?
------------------

=> We look at frequent phrases, and rules

- Find all **highly frequent** phrases across a set of reviews (e.g. `fish tacos`) => this can help identify the targets of different sentiments.
- Filter these highly frequent phrases by rules like **occurs right after a sentiment word**

=> `... great fish tacos ...` means that **fish tacos** is a likely target of sentiment, since we know **great** is a sentiment word.

Let's say we already know the important aspects of a piece of text. For example, if we are analyzing restaurant reviews, we know that aspects we will come across include **food**, **decor**, **service**, **value**, ...

Then we can train our classifier to assign an aspect to a given sentence or phrase.

"Given this sentence, is it talking about **food** or **decor** or ..."

=> This only applies to text where we KNOW what we will come across.

So overall, our flow could look like:

**Text (e.g. reviews)** --> **Text extractor (extract sentences/phrases)** --> **Sentiment Classifier (assign a sentiment to each sentence/phrase)** --> **Aspect Extractor (assign an aspect to each sentence/phrase)** --> **Aggregator** --> **Final Summary**

##Conditional Models

Naive Bayes Classifiers use a joint probability model. We evaluate probabilities P( d, c ) and try to maximize this joint likelihood.

=> maximizing P( text, class ) 

rather than a conditional probability model

-> maximizing P( class | text )

If we instead try to maximize the conditional probability of P( class | text ), we can achieve higher accuracy in our classifier.

A **conditional** model gives probabilities **P( c | d )**. It takes the data as given and models only the  conditional probability of the class.

##MaxEnt Classifiers (Maximum Entropy Classifiers)

We define a **feature** as an elementary piece of evidence that links aspects of what we observe ( **d** ), with a category ( **c** ) that we want to predict.

So a feature is a function that maps from the space of **classes** and **data** onto a **Real Number** (it has a bounded, real value).

&fnof;: C x D --> R

Models will assign a **weight** to each feature:

- A **positive** weight votes that the configuration is likely correct
- A **negative** weight votes that the configuration is likely incorrect 

What do features look like?
---------------------------
Here is an example feature:

- &fnof;<sub>1</sub>(c,d) &equiv; [ c = LOCATION & w<sub>-1</sub>="in" & isCapitalized(w) ]

This feature picks out from the data cases where the **class** is **LOCATION**, the previous word is "in" and the current word is capitalized.

This feature would match the following scenarios:

- class = LOCATION, data = "in Quebec"
- class = LOCATION, data = "in Arcadia"

Another example feature:

- &fnof;<sub>2</sub>(c,d) &equiv; [ c = DRUG & ends(w, "c") ]

This feature picks out from the data cases where the **class** is **DRUG** and the current word ends with the letter **c**.

This feature would match:

- class = DRUG, data = "taking Zantac"

Features generally use both the **bag of words**, as we saw with the Naive-Bayes Classifier, as well as looking at adjacent words (like the example features above).

###Feature-Based Linear Classifiers

Feature-Based Linear Classifiers:

- Are a linear function from feature sets {&fnof;<sub>i</sub>} to classes {c}.
- Assign a weight &lambda;<sub>i</sub> to each feature &fnof;<sub>i</sub>
- We consider each class for an observed datum **d**
- For a pair **(c,d)**, features vote with their weights:

  **vote(c)** = &Sigma; &lambda;<sub>i</sub>&fnof;<sub>i</sub>(c,d)

- Choose the class **c** which maximizes **vote(c)**

As you can see in the equation above, the vote is just a weighted sum of the features; each feature has its own weight. So we try to find the class that maximizes the weighted sum of all the features.

MaxEnt Models make a probabilistic model from the linear combination &Sigma; &lambda;<sub>i</sub>&fnof;<sub>i</sub>(c,d).

Since the weights can be negative values, we need to convert them to positive values since we want to calculating a non-negative probability for a given class. So we use the value as such:

**exp &Sigma; &lambda;<sub>i</sub>&fnof;<sub>i</sub>(c,d)**

This way we will always have a positive value.

We make this value into a probability by dividing by the sum of the probabilities of all classes:

**[ exp &Sigma; &lambda;<sub>i</sub>&fnof;<sub>i</sub>(c,d) ] / [ &Sigma;<sub>C</sub> exp &Sigma; &lambda;<sub>i</sub>&fnof;<sub>i</sub>(c,d) ]**


##Named Entity Recognition

**Named Entity Recognition (NER)** is the task of extracting entities (people, organizations, dates, etc.) from text.

###Machine-Learning sequence model approach to NER

####Training

- Collect a set of representative Training Documents
- Label each token for its entity class, or Other (O) if no match
- Design feature extractors appropriate to the text and classes
- Train a sequence classifier to predict the labels from the data

####Testing

- Get a set of testing documents
- Run the model on the document to label each token
- Output the recognized entities



