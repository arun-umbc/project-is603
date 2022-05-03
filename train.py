import numpy
import xgboost
from keras import layers, models
from keras.preprocessing import text, sequence
from pandas import read_csv
from sklearn import model_selection, preprocessing, metrics, naive_bayes, linear_model, svm, ensemble
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from tensorflow.keras import optimizers

CATEGORIES = ['automobile', 'crime', 'entertainment', 'health', 'politics', 'sport']


def load_data_set_from_csv():
    """ load the data set """
    df = read_csv('data/data_output.csv')
    df = df.dropna()
    return df


def split_data(train_df):
    """ split the data set into training and validation data sets """
    train_x, valid_x, train_y, valid_y = model_selection.train_test_split(train_df['text'], train_df['category'],
                                                                          test_size=0.3)
    return train_x, valid_x, train_y, valid_y


def encode_label_data(train_y, valid_y):
    """ label encode the target variable """
    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    valid_y = encoder.fit_transform(valid_y)
    return train_y, valid_y


def encode_label_data_for_accuracy_test(valid_y):
    """ label encode the target variable """
    encoder = preprocessing.LabelEncoder()
    valid_y = encoder.fit_transform(valid_y)
    return valid_y


def create_vector_object(train_df):
    """ create a count vector object """
    count_vector = CountVectorizer(analyzer=lambda x: x.split(), token_pattern=r'\w{1,}', max_features=9000,
                                   ngram_range=(2, 3))
    count_vector.fit(train_df['text'])
    return count_vector


def transform_train_data(count_vector, train_x, valid_x):
    """ transform the training and validation data using count vector object """
    x_train_count = count_vector.transform(train_x)
    x_valid_count = count_vector.transform(valid_x)
    return x_train_count, x_valid_count


def create_word_tf_idf(train_df, train_x, valid_x):
    """ word level tf-idf """
    tfidf_vector = TfidfVectorizer(analyzer='word', max_features=5000)
    tfidf_vector.fit(train_df['text'])
    x_train_tfidf = tfidf_vector.transform(train_x)
    x_valid_tfidf = tfidf_vector.transform(valid_x)
    return x_train_tfidf, x_valid_tfidf


def create_ngram_tf_idf(train_df, train_x, valid_x):
    """ ngram level tf-idf """
    tfidf_vector_ngram = TfidfVectorizer(analyzer=lambda x: x.split(), ngram_range=(1, 2), max_features=7000)
    tfidf_vector_ngram.fit(train_df['text'])
    x_train_tfidf_ngram = tfidf_vector_ngram.transform(train_x)
    x_valid_tfidf_ngram = tfidf_vector_ngram.transform(valid_x)
    return x_train_tfidf_ngram, x_valid_tfidf_ngram


def create_character_tf_idf(train_df, train_x, valid_x):
    """ characters level tf-idf """
    tfidf_vector_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(1, 2),
                                               max_features=5000)
    tfidf_vector_ngram_chars.fit(train_df['text'])
    x_train_tfidf_ngram_chars = tfidf_vector_ngram_chars.transform(train_x)
    x_valid_tfidf_ngram_chars = tfidf_vector_ngram_chars.transform(valid_x)
    return x_train_tfidf_ngram_chars, x_valid_tfidf_ngram_chars


def load_pre_trained_embedded_vector():
    """ load the pre-trained word-embedding vectors """
    embeddings_index = {}
    for i, line in enumerate(open('model_files/wiki.hi.vec', encoding='utf-8')):
        try:
            values = line.split()
            embeddings_index[values[0]] = numpy.asarray(values[1:], dtype='float32')
        except ValueError:
            continue
    return embeddings_index


def create_tokenizer(train_df):
    """ create a tokenizer """
    token = text.Tokenizer()
    token.fit_on_texts(train_df['text'])
    word_index = token.word_index
    return token, word_index


def convert_text_to_token(train_x, valid_x, token):
    """ convert text to sequence of tokens and pad them to ensure equal length vectors """
    train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=70)
    valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=70)
    return train_seq_x, valid_seq_x


def map_token_embedding(word_index, embeddings_index):
    """ create token-embedding mapping """
    embedding_matrix = numpy.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def create_cnn(word_index, embedding_matrix):
    """ creating a Convolutional neural network"""

    # Add an Input Layer
    input_layer = layers.Input((70,))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(
        input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the convolutional Layer
    conv_layer = layers.Convolution1D(100, 3, activation="relu")(embedding_layer)

    # Add the pooling Layer
    pooling_layer = layers.GlobalMaxPool1D()(conv_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(50, activation="relu")(pooling_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')

    return model


def create_rnn_lstm(word_index, embedding_matrix):
    # Add an Input Layer
    input_layer = layers.Input((70,))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(
        input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the LSTM Layer
    lstm_layer = layers.LSTM(100)(embedding_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(50, activation="relu")(lstm_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')

    return model


def create_rnn_gru(word_index, embedding_matrix):
    # Add an Input Layer
    input_layer = layers.Input((70,))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(
        input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the GRU Layer
    lstm_layer = layers.GRU(100)(embedding_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(50, activation="relu")(lstm_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')

    return model


def create_bi_rnn(word_index, embedding_matrix):
    # Add an Input Layer
    input_layer = layers.Input((70,))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(
        input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the LSTM Layer
    lstm_layer = layers.Bidirectional(layers.GRU(100))(embedding_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(50, activation="relu")(lstm_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')

    return model


def train_model(classifier, feature_vector_train, label, feature_vector_valid, valid_y, name, is_neural_net=False):
    """ model training """

    # fit the training data set on the classifier
    classifier.fit(feature_vector_train, label)

    # predict the labels on validation data set
    predictions = classifier.predict(feature_vector_valid)

    if is_neural_net:
        predictions = predictions.argmax(axis=-1)

    return metrics.accuracy_score(predictions, valid_y)


def test_cnn_accuracy(word_index, embedding_matrix, train_seq_x, train_y, valid_seq_x, valid_y):
    classifier = create_cnn(word_index, embedding_matrix)
    accuracy = train_model(classifier, train_seq_x, train_y, valid_seq_x, valid_y, is_neural_net=True, name='cnn')
    print("CNN, Word Embeddings", accuracy)


def test_rnn_lstm_accuracy(word_index, embedding_matrix, train_seq_x, train_y, valid_seq_x, valid_y):
    classifier = create_rnn_lstm(word_index, embedding_matrix)
    accuracy = train_model(classifier, train_seq_x, train_y, valid_seq_x, valid_y, is_neural_net=True, name='cnn')
    print("RNN_LSTM, Word Embeddings", accuracy)


def test_rnn_gru_accuracy(word_index, embedding_matrix, train_seq_x, train_y, valid_seq_x, valid_y):
    classifier = create_rnn_gru(word_index, embedding_matrix)
    accuracy = train_model(classifier, train_seq_x, train_y, valid_seq_x, valid_y, is_neural_net=True, name='cnn')
    print("RNN_GRU, Word Embeddings", accuracy)


def test_rnn_bi_accuracy(word_index, embedding_matrix, train_seq_x, train_y, valid_seq_x, valid_y):
    classifier = create_bi_rnn(word_index, embedding_matrix)
    accuracy = train_model(classifier, train_seq_x, train_y, valid_seq_x, valid_y, is_neural_net=True, name='cnn')
    print("RNN_BI, Word Embeddings", accuracy)


def test_nb_accuracy(x_train_count, x_train_tfidf, x_train_tfidf_ngram, x_train_tfidf_ngram_chars, train_y,
                     x_valid_count, x_valid_tfidf, x_valid_tfidf_ngram, x_valid_tfidf_ngram_chars, valid_y):
    # Naive Bayes on Count Vectors
    accuracy = train_model(naive_bayes.MultinomialNB(), x_train_count, train_y, x_valid_count, valid_y,
                           name="count_vector")
    print("NB, Count Vectors: ", accuracy)

    # Naive Bayes on Word Level TF IDF Vectors
    accuracy = train_model(naive_bayes.MultinomialNB(), x_train_tfidf, train_y, x_valid_tfidf, valid_y,
                           name="word_level")
    print("NB, WordLevel TF-IDF: ", accuracy)

    # Naive Bayes on Ngram Level TF IDF Vectors
    accuracy = train_model(naive_bayes.MultinomialNB(), x_train_tfidf_ngram, train_y, x_valid_tfidf_ngram, valid_y,
                           name="n_gram_vector")
    print("NB, N-Gram Vectors: ", accuracy)

    # Naive Bayes on Character Level TF IDF Vectors
    accuracy = train_model(naive_bayes.MultinomialNB(), x_train_tfidf_ngram_chars, train_y, x_valid_tfidf_ngram_chars,
                           valid_y, name="char_level_vector")
    print("NB, CharLevel Vectors: ", accuracy)


def test_logistic_accuracy(x_train_count, x_train_tfidf, x_train_tfidf_ngram, x_train_tfidf_ngram_chars, train_y,
                           x_valid_count, x_valid_tfidf, x_valid_tfidf_ngram, x_valid_tfidf_ngram_chars, valid_y):
    # Logistic regression on Count Vectors
    accuracy = train_model(linear_model.LogisticRegression(), x_train_count, train_y, x_valid_count, valid_y,
                           name="count_vector")
    print("LR, Count Vectors: ", accuracy)

    # Logistic regression on Word Level TF IDF Vectors
    accuracy = train_model(linear_model.LogisticRegression(), x_train_tfidf, train_y, x_valid_tfidf, valid_y,
                           name="word_level")
    print("LR, WordLevel TF-IDF: ", accuracy)

    # Logistic regression on Ngram Level TF IDF Vectors
    accuracy = train_model(linear_model.LogisticRegression(), x_train_tfidf_ngram, train_y, x_valid_tfidf_ngram, valid_y,
                           name="n_gram_vector")
    print("LR, N-Gram Vectors: ", accuracy)

    # Logistic regression on Character Level TF IDF Vectors
    accuracy = train_model(linear_model.LogisticRegression(), x_train_tfidf_ngram_chars, train_y, x_valid_tfidf_ngram_chars,
                           valid_y, name="char_level_vector")
    print("LR, CharLevel Vectors: ", accuracy)


def test_svm_accuracy(x_train_count, x_train_tfidf, x_train_tfidf_ngram, x_train_tfidf_ngram_chars, train_y,
                      x_valid_count, x_valid_tfidf, x_valid_tfidf_ngram, x_valid_tfidf_ngram_chars, valid_y):
    # Support Vector Machine on Count Vectors
    accuracy = train_model(svm.SVC(), x_train_count, train_y, x_valid_count, valid_y,
                           name="count_vector")
    print("SVM, Count Vectors: ", accuracy)

    # Support Vector Machine on Word Level TF IDF Vectors
    accuracy = train_model(svm.SVC(), x_train_tfidf, train_y, x_valid_tfidf, valid_y,
                           name="word_level")
    print("SVM, WordLevel TF-IDF: ", accuracy)

    # Support Vector Machine on Ngram Level TF IDF Vectors
    accuracy = train_model(svm.SVC(), x_train_tfidf_ngram, train_y, x_valid_tfidf_ngram, valid_y,
                           name="n_gram_vector")
    print("SVM, N-Gram Vectors: ", accuracy)

    # Support Vector Machine on Character Level TF IDF Vectors
    accuracy = train_model(svm.SVC(), x_train_tfidf_ngram_chars, train_y, x_valid_tfidf_ngram_chars,
                           valid_y, name="char_level_vector")
    print("SVM, CharLevel Vectors: ", accuracy)


def test_rf_accuracy(x_train_count, x_train_tfidf, x_train_tfidf_ngram, x_train_tfidf_ngram_chars, train_y,
                     x_valid_count, x_valid_tfidf, x_valid_tfidf_ngram, x_valid_tfidf_ngram_chars, valid_y):
    # Random Forest on Count Vectors
    accuracy = train_model(ensemble.RandomForestClassifier(), x_train_count, train_y, x_valid_count, valid_y,
                           name="count_vector")
    print("RF, Count Vectors: ", accuracy)

    # Random Forest on Word Level TF IDF Vectors
    accuracy = train_model(ensemble.RandomForestClassifier(), x_train_tfidf, train_y, x_valid_tfidf, valid_y,
                           name="word_level")
    print("RF, WordLevel TF-IDF: ", accuracy)

    # Random Forest on Ngram Level TF IDF Vectors
    accuracy = train_model(ensemble.RandomForestClassifier(), x_train_tfidf_ngram, train_y, x_valid_tfidf_ngram, valid_y,
                           name="n_gram_vector")
    print("RF, N-Gram Vectors: ", accuracy)

    # Random Forest on Character Level TF IDF Vectors
    accuracy = train_model(ensemble.RandomForestClassifier(), x_train_tfidf_ngram_chars, train_y, x_valid_tfidf_ngram_chars,
                           valid_y, name="char_level_vector")
    print("RF, CharLevel Vectors: ", accuracy)


def test_xg_accuracy(x_train_count, x_train_tfidf, x_train_tfidf_ngram, x_train_tfidf_ngram_chars, train_y,
                     x_valid_count, x_valid_tfidf, x_valid_tfidf_ngram, x_valid_tfidf_ngram_chars, valid_y):
    # XGBoost on Count Vectors
    accuracy = train_model(xgboost.XGBClassifier(), x_train_count, train_y, x_valid_count, valid_y,
                           name="count_vector")
    print("XG, Count Vectors: ", accuracy)

    # XGBoost on Word Level TF IDF Vectors
    accuracy = train_model(xgboost.XGBClassifier(), x_train_tfidf, train_y, x_valid_tfidf, valid_y,
                           name="word_level")
    print("XG, WordLevel TF-IDF: ", accuracy)

    # XGBoost on Ngram Level TF IDF Vectors
    accuracy = train_model(xgboost.XGBClassifier(), x_train_tfidf_ngram, train_y, x_valid_tfidf_ngram, valid_y,
                           name="n_gram_vector")
    print("XG, N-Gram Vectors: ", accuracy)

    # XGBoost on Character Level TF IDF Vectors
    accuracy = train_model(xgboost.XGBClassifier(), x_train_tfidf_ngram_chars, train_y, x_valid_tfidf_ngram_chars,
                           valid_y, name="char_level_vector")
    print("XG, CharLevel Vectors: ", accuracy)
