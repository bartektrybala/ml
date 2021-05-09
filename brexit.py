import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


def prediction(string):
    filepath = 'brexit.txt'

    df = pd.read_csv(filepath, names=['label', 'sentence'], sep='\t')

    first_column = df.pop('sentence')
    df.insert(0, 'sentence', first_column)

    col = ['sentence', 'label']
    df = df[col]
    df = df[pd.notnull(df['label'])]
    df.columns = ['sentence', 'label']

    df['category_id'] = df['label'].factorize()[0]
    category_id_df = df[['label', 'category_id']].drop_duplicates().sort_values('category_id')
    category_to_id = dict(category_id_df.values)

    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2',
                        encoding='latin-1', ngram_range=(1, 2), stop_words='english')

    features = tfidf.fit_transform(df.sentence).toarray()
    labels = df.category_id

    # najczęściej występowane frazy
    N = 2
    for label, category_id in sorted(category_to_id.items()):
        features_chi2 = chi2(features, labels == category_id)
        indices = np.argsort(features_chi2[0])
        feature_names = np.array(tfidf.get_feature_names())[indices]
        unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
        bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
        #print("# '{}':".format(label))
        #print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
        #print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))
        #print('\n ------------------------------- \n')

    X_train, X_test, y_train, y_test = train_test_split(df['sentence'], df['label'], random_state = 0)
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    clf = MultinomialNB().fit(X_train_tfidf, y_train)

    return clf.predict(count_vect.transform(([string])))



