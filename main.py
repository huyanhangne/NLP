## for data
import json
import pandas as pd
import numpy as np
from sklearn import metrics, manifold
## for processing
import re
import nltk
## for plotting
import matplotlib.pyplot as plt
import seaborn as sns
## for w2v
import gensim
import gensim.downloader as gensim_api
## for bert
import transformers

lst_dics = []
with open('F:\laodong-kinhte\923924.json', mode='r', errors='ignore') as json_file:
    for dic in json_file: lst_dics.append(json.loads(dic))
## print the first one
lst_dics[0]

## create dtf
dtf = pd.DataFrame(lst_dics)
## filter categories
dtf = dtf[ dtf["category"].isin(['ENTERTAINMENT','POLITICS','TECH'])        ][["category","headline"]]
## rename columns
dtf = dtf.rename(columns={"category":"y", "headline":"text"})
## print 5 random rows
dtf.sample(5)

'''
Preprocess a string.
:parameter
    :param text: string - name of column containing text
    :param lst_stopwords: list - list of stopwords to remove
    :param flg_stemm: bool - whether stemming is to be applied
    :param flg_lemm: bool - whether lemmitisation is to be applied
:return
    cleaned text
'''


def utils_preprocess_text(text, flg_stemm=False, flg_lemm=True, lst_stopwords=None):
    ## clean (convert to lowercase and remove punctuations and  characters and the    strip)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())

    ## Tokenize (convert from string to list)
    lst_text = text.split()
    ## remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in
    lst_stopwords]

    ## Stemming (remove -ing, -ly, ...)
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
    lst_text = [ps.stem(word) for word in lst_text]

    ## Lemmatisation (convert the word into root word)
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
    lst_text = [lem.lemmatize(word) for word in lst_text]

    ## back to string from list

    text = " ".join(lst_text)
    return text

    lst_stopwords = nltk.corpus.stopwords.words("english")
    lst_stopwords

    dtf["text_clean"] = dtf["text"].apply(lambda x:utils_preprocess_text(x, flg_stemm=False, flg_lemm=True,lst_stopwords=lst_stopwords))
    dtf.head()

    nlp.most_similar(["obama"], topn=3)

    ## Function to apply
    def get_similar_words(lst_words, top, nlp):
        lst_out = lst_words
        for tupla in nlp.most_similar(lst_words, topn=top):
            lst_out.append(tupla[0])
        return list(set(lst_out))

    ## Create Dictionary {category:[keywords]}
    dic_clusters = {}
    dic_clusters["ENTERTAINMENT"] = get_similar_words(['celebrity', 'cinema', 'movie', 'music'],
                                                      top=30, nlp=nlp)
    dic_clusters["POLITICS"] = get_similar_words(['gop', 'clinton', 'president', 'obama', 'republican']
                                                 , top=30, nlp=nlp)
    dic_clusters["TECH"] = get_similar_words(['amazon', 'android', 'app', 'apple', 'facebook',
                                              'google', 'tech'],
                                             top=30, nlp=nlp)
    ## print some
    for k, v in dic_clusters.items():
        print(k, ": ", v[0:5], "...", len(v))

    ## word embedding
    tot_words = [word for v in dic_clusters.values() for word in v]
    X = nlp[tot_words]

    ## pca
    pca = manifold.TSNE(perplexity=40, n_components=2, init='pca')
    X = pca.fit_transform(X)

    ## create dtf
    dtf = pd.DataFrame()
    for k, v in dic_clusters.items():
        size = len(dtf) + len(v)
        dtf_group = pd.DataFrame(X[len(dtf):size], columns=["x", "y"],
                                 index=v)
        dtf_group["cluster"] = k
        dtf = dtf.append(dtf_group)

    ## plot
    fig, ax = plt.subplots()
    sns.scatterplot(data=dtf, x="x", y="y", hue="cluster", ax=ax)
    ax.legend().texts[0].set_text(None)
    ax.set(xlabel=None, ylabel=None, xticks=[], xticklabels=[],
           yticks=[], yticklabels=[])
    for i in range(len(dtf)):
        ax.annotate(dtf.index[i],
                    xy=(dtf["x"].iloc[i], dtf["y"].iloc[i]),
                    xytext=(5, 2), textcoords='offset points',
                    ha='right', va='bottom')

    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    nlp = transformers.TFBertModel.from_pretrained('bert-base-uncased')

    txt = "river bank"
    ## tokenize
    idx = tokenizer.encode(txt)
    print("tokens:", tokenizer.convert_ids_to_tokens(idx))
    print("ids   :", tokenizer.encode(txt))
    ## word embedding
    idx = np.array(idx)[None, :]
    embedding = nlp(idx)
    print("shape:", embedding[0][0].shape)
    ## vector of the second input word
    embedding[0][0][2]

    ## function to apply
    def utils_bert_embedding(txt, tokenizer, nlp):
        idx = tokenizer.encode(txt)
        idx = np.array(idx)[None, :]
        embedding = nlp(idx)
        X = np.array(embedding[0][0][1:-1])
        return X

    ## create list of news vector
    lst_mean_vecs = [utils_bert_embedding(txt, tokenizer, nlp).mean(0)
                     for txt in dtf["text_clean"]]
    ## create the feature matrix (n news x 768)
    X = np.array(lst_mean_vecs)

    dic_y = {k: utils_bert_embedding(v, tokenizer, nlp).mean(0) for k, v
             in dic_clusters.items()}

    # --- Model Algorithm ---#
    ## compute cosine similarities
    similarities = np.array(
        [metrics.pairwise.cosine_similarity(X, y).T.tolist()[0]
         for y in dic_y.values()]
    ).T
    ## adjust and rescale
    labels = list(dic_y.keys())
    for i in range(len(similarities)):
        ### assign randomly if there is no similarity
        if sum(similarities[i]) == 0:
            similarities[i] = [0] * len(labels)
            similarities[i][np.random.choice(range(len(labels)))] = 1
        ### rescale so they sum = 1
        similarities[i] = similarities[i] / sum(similarities[i])

    ## classify the label with highest similarity score
    predicted_prob = similarities
    predicted = [labels[np.argmax(pred)] for pred in predicted_prob]

    y_test = dtf["y"].values
    classes = np.unique(y_test)
    y_test_array = pd.get_dummies(y_test, drop_first=False).values

    ## Accuracy, Precision, Recall
    accuracy = metrics.accuracy_score(y_test, predicted)
    auc = metrics.roc_auc_score(y_test, predicted_prob,
                                multi_class="ovr")
    print("Accuracy:", round(accuracy, 2))
    print("Auc:", round(auc, 2))
    print("Detail:")
    print(metrics.classification_report(y_test, predicted))

    ## Plot confusion matrix
    cm = metrics.confusion_matrix(y_test, predicted)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues,
                cbar=False)
    ax.set(xlabel="Pred", ylabel="True", xticklabels=classes,
           yticklabels=classes, title="Confusion matrix")
    plt.yticks(rotation=0)
    fig, ax = plt.subplots(nrows=1, ncols=2)

    ## Plot roc
    for i in range(len(classes)):
        fpr, tpr, thresholds = metrics.roc_curve(y_test_array[:, i],
                                                 predicted_prob[:, i])
        ax[0].plot(fpr, tpr, lw=3,
                   label='{0} (area={1:0.2f})'.format(classes[i],
                                                      metrics.auc(fpr, tpr))
                   )
    ax[0].plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
    ax[0].set(xlim=[-0.05, 1.0], ylim=[0.0, 1.05],
              xlabel='False Positive Rate',
              ylabel="True Positive Rate (Recall)",
              title="Receiver operating characteristic")
    ax[0].legend(loc="lower right")
    ax[0].grid(True)

    ## Plot precision-recall curve
    for i in range(len(classes)):
        precision, recall, thresholds = metrics.precision_recall_curve(
            y_test_array[:, i], predicted_prob[:, i])
        ax[1].plot(recall, precision, lw=3,
                   label='{0} (area={1:0.2f})'.format(classes[i],
                                                      metrics.auc(recall, precision))
                   )
    ax[1].set(xlim=[0.0, 1.05], ylim=[0.0, 1.05], xlabel='Recall',
              ylabel="Precision", title="Precision-Recall curve")
    ax[1].legend(loc="best")
    ax[1].grid(True)
    plt.show()

    i = 7
    txt_instance = dtf["text_clean"].iloc[i]
    print("True:", y_test[i], "--> Pred:", predicted[i], "|Similarity: ", round(np.max(predicted_prob[i]),2))
    print(txt_instance)

    ## create embedding Matrix
    y = np.concatenate([embedding_bert(v, tokenizer, nlp) for v in
                        dic_clusters.values()])
    X = embedding_bert(txt_instance, tokenizer,
                       nlp).mean(0).reshape(1, -1)
    M = np.concatenate([y, X])

    ## pca
    pca = manifold.TSNE(perplexity=40, n_components=2, init='pca')
    M = pca.fit_transform(M)
    y, X = M[:len(y)], M[len(y):]

    ## create dtf clusters
    dtf = pd.DataFrame()
    for k, v in dic_clusters.items():
        size = len(dtf) + len(v)
        dtf_group = pd.DataFrame(y[len(dtf):size], columns=["x", "y"],
                                 index=v)
        dtf_group["cluster"] = k
        dtf = dtf.append(dtf_group)

    ## plot clusters
    fig, ax = plt.subplots()
    sns.scatterplot(data=dtf, x="x", y="y", hue="cluster", ax=ax)
    ax.legend().texts[0].set_text(None)
    ax.set(xlabel=None, ylabel=None, xticks=[], xticklabels=[],
           yticks=[], yticklabels=[])
    for i in range(len(dtf)):
        ax.annotate(dtf.index[i],
                    xy=(dtf["x"].iloc[i], dtf["y"].iloc[i]),
                    xytext=(5, 2), textcoords='offset points',
                    ha='right', va='bottom')

    ## add txt_instance
    ax.scatter(x=X[0][0], y=X[0][1], c="red", linewidth=10)
    ax.annotate("x", xy=(X[0][0], X[0][1]),
                ha='center', va='center', fontsize=25)


## calculate similarity
sim_matrix = metrics.pairwise.cosine_similarity(X, y)

## add top similarity
for row in range(sim_matrix.shape[0]):
    ### sorted {keyword:score}
    dic_sim = {n: sim_matrix[row][n] for n in
               range(sim_matrix.shape[1])}
    dic_sim = {k: v for k, v in sorted(dic_sim.items(),
                                       key=lambda item: item[1], reverse=True)}
    ### plot lines
    for k in dict(list(dic_sim.items())[0:5]).keys():
        p1 = [X[row][0], X[row][1]]
        p2 = [y[k][0], y[k][1]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], c="red", alpha=0.5)
plt.show()

## create embedding Matrix
y = np.concatenate([embedding_bert(v, tokenizer, nlp) for v in
                    dic_clusters.values()])
X = embedding_bert(txt_instance, tokenizer,
                   nlp).mean(0).reshape(1,-1)
M = np.concatenate([y,X])

## pca
pca = manifold.TSNE(perplexity=40, n_components=2, init='pca')
M = pca.fit_transform(M)
y, X = M[:len(y)], M[len(y):]

## create dtf clusters
dtf = pd.DataFrame()
for k,v in dic_clusters.items():
    size = len(dtf) + len(v)
    dtf_group = pd.DataFrame(y[len(dtf):size], columns=["x","y"],
                             index=v)
    dtf_group["cluster"] = k
    dtf = dtf.append(dtf_group)

## add txt_instance
tokens = tokenizer.convert_ids_to_tokens(
               tokenizer.encode(txt_instance))[1:-1]
dtf = pd.DataFrame(X, columns=["x","y"], index=tokens)
dtf = dtf[~dtf.index.str.contains("#")]
dtf = dtf[dtf.index.str.len() > 1]
X = dtf.values
ax.scatter(x=dtf["x"], y=dtf["y"], c="red")
for i in range(len(dtf)):
     ax.annotate(dtf.index[i],
                 xy=(dtf["x"].iloc[i],dtf["y"].iloc[i]),
                 xytext=(5,2), textcoords='offset points',
                 ha='right', va='bottom')

## calculate similarity
sim_matrix = metrics.pairwise.cosine_similarity(X, y)

## add top similarity
for row in range(sim_matrix.shape[0]):
    ### sorted {keyword:score}
    dic_sim = {n:sim_matrix[row][n] for n in
               range(sim_matrix.shape[1])}
    dic_sim = {k:v for k,v in sorted(dic_sim.items(),
                key=lambda item:item[1], reverse=True)}
    ### plot lines
    for k in dict(list(dic_sim.items())[0:5]).keys():
        p1 = [X[row][0], X[row][1]]
        p2 = [y[k][0], y[k][1]]
        ax.plot([p1[0],p2[0]], [p1[1],p2[1]], c="red", alpha=0.5)
plt.show()
