from train import load_data_set_from_csv, split_data, encode_label_data, create_vector_object, transform_train_data, \
    create_word_tf_idf, create_ngram_tf_idf, create_character_tf_idf, test_nb_accuracy, test_logistic_accuracy, \
    test_svm_accuracy, test_rf_accuracy, test_xg_accuracy, test_cnn_accuracy, load_pre_trained_embedded_vector, \
    create_tokenizer, convert_text_to_token, map_token_embedding, test_rnn_lstm_accuracy, test_rnn_gru_accuracy, \
    test_rnn_bi_accuracy


def prep_model():
    train_df = load_data_set_from_csv()
    train_x, valid_x, train_y, valid_y = split_data(train_df)
    train_y, valid_y = encode_label_data(train_y, valid_y)
    count_vector = create_vector_object(train_df)
    x_train_count, x_valid_count = transform_train_data(count_vector, train_x, valid_x)
    x_train_tfidf, x_valid_tfidf = create_word_tf_idf(train_df, train_x, valid_x)
    x_train_tfidf_ngram, x_valid_tfidf_ngram = create_ngram_tf_idf(train_df, train_x, valid_x)
    x_train_tfidf_ngram_chars, x_valid_tfidf_ngram_chars = create_character_tf_idf(train_df, train_x, valid_x)
    return x_train_count, x_train_tfidf, x_train_tfidf_ngram, x_train_tfidf_ngram_chars, train_y, x_valid_count, \
        x_valid_tfidf, x_valid_tfidf_ngram, x_valid_tfidf_ngram_chars, valid_y


def run_nb():
    x_train_count, x_train_tfidf, x_train_tfidf_ngram, x_train_tfidf_ngram_chars, train_y, x_valid_count, \
     x_valid_tfidf, x_valid_tfidf_ngram, x_valid_tfidf_ngram_chars, valid_y = prep_model()
    test_nb_accuracy(x_train_count, x_train_tfidf, x_train_tfidf_ngram, x_train_tfidf_ngram_chars, train_y,
                     x_valid_count, x_valid_tfidf, x_valid_tfidf_ngram, x_valid_tfidf_ngram_chars, valid_y)


def run_logistic():
    x_train_count, x_train_tfidf, x_train_tfidf_ngram, x_train_tfidf_ngram_chars, train_y, x_valid_count, \
     x_valid_tfidf, x_valid_tfidf_ngram, x_valid_tfidf_ngram_chars, valid_y = prep_model()
    test_logistic_accuracy(x_train_count, x_train_tfidf, x_train_tfidf_ngram, x_train_tfidf_ngram_chars, train_y,
                           x_valid_count, x_valid_tfidf, x_valid_tfidf_ngram, x_valid_tfidf_ngram_chars, valid_y)


def run_svm():
    x_train_count, x_train_tfidf, x_train_tfidf_ngram, x_train_tfidf_ngram_chars, train_y, x_valid_count, \
     x_valid_tfidf, x_valid_tfidf_ngram, x_valid_tfidf_ngram_chars, valid_y = prep_model()
    test_svm_accuracy(x_train_count, x_train_tfidf, x_train_tfidf_ngram, x_train_tfidf_ngram_chars, train_y,
                      x_valid_count, x_valid_tfidf, x_valid_tfidf_ngram, x_valid_tfidf_ngram_chars, valid_y)


def run_rf():
    x_train_count, x_train_tfidf, x_train_tfidf_ngram, x_train_tfidf_ngram_chars, train_y, x_valid_count, \
     x_valid_tfidf, x_valid_tfidf_ngram, x_valid_tfidf_ngram_chars, valid_y = prep_model()
    test_rf_accuracy(x_train_count, x_train_tfidf, x_train_tfidf_ngram, x_train_tfidf_ngram_chars, train_y,
                     x_valid_count, x_valid_tfidf, x_valid_tfidf_ngram, x_valid_tfidf_ngram_chars, valid_y)


def run_xg_boost():
    x_train_count, x_train_tfidf, x_train_tfidf_ngram, x_train_tfidf_ngram_chars, train_y, x_valid_count, \
     x_valid_tfidf, x_valid_tfidf_ngram, x_valid_tfidf_ngram_chars, valid_y = prep_model()
    test_xg_accuracy(x_train_count, x_train_tfidf, x_train_tfidf_ngram, x_train_tfidf_ngram_chars, train_y,
                     x_valid_count, x_valid_tfidf, x_valid_tfidf_ngram, x_valid_tfidf_ngram_chars, valid_y)


def prep_nn_model():
    embeddings_index = load_pre_trained_embedded_vector()
    train_df = load_data_set_from_csv()
    train_x, valid_x, train_y, valid_y = split_data(train_df)
    train_y, valid_y = encode_label_data(train_y, valid_y)
    token, word_index = create_tokenizer(train_df)
    train_seq_x, valid_seq_x = convert_text_to_token(train_x, valid_x, token)
    embedding_matrix = map_token_embedding(word_index, embeddings_index)
    return word_index, embedding_matrix, train_seq_x, train_y, valid_seq_x, valid_y


def run_cnn():
    word_index, embedding_matrix, train_seq_x, train_y, valid_seq_x, valid_y = prep_nn_model()
    test_cnn_accuracy(word_index, embedding_matrix, train_seq_x, train_y, valid_seq_x, valid_y)


def run_rnn_lstm():
    word_index, embedding_matrix, train_seq_x, train_y, valid_seq_x, valid_y = prep_nn_model()
    test_rnn_lstm_accuracy(word_index, embedding_matrix, train_seq_x, train_y, valid_seq_x, valid_y)


def run_rnn_gru():
    word_index, embedding_matrix, train_seq_x, train_y, valid_seq_x, valid_y = prep_nn_model()
    test_rnn_gru_accuracy(word_index, embedding_matrix, train_seq_x, train_y, valid_seq_x, valid_y)


def run_rnn_bi():
    word_index, embedding_matrix, train_seq_x, train_y, valid_seq_x, valid_y = prep_nn_model()
    test_rnn_bi_accuracy(word_index, embedding_matrix, train_seq_x, train_y, valid_seq_x, valid_y)


run_nb()
run_logistic()
run_svm()
run_rf()
run_xg_boost()
run_cnn()
run_rnn_lstm()
run_rnn_gru()
run_rnn_bi()
