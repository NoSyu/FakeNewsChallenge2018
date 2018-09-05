from custom_vectorizer.gru_utils import *

class GRUVectorizer:

    def __init__(self, BaseEstimator=None, TransformerMixin=None):
        self.baseEstimator = BaseEstimator
        self.transformerMixin = TransformerMixin
        self.param_dict = {
            "MAX_NB_WORDS": 50000,  # size of the vocabulary

            # sequence lengths
            "MAX_SEQ_LENGTH": 100,

            # embedding specific values
            "EMBEDDING_DIM": 50,  # dimension of the GloVe embeddings
            "GLOVE_ZIP_FILE": "glove.twitter.27B.zip",
            "GLOVE_FILE": "glove.twitter.27B.50d.txt",

            # embedding file names
            "EMBEDDING_FILE": "single_flat_LSTM_50d_100_embedding.npy",

            # vocab file names
            "VOCAB_FILE": "single_flat_LSTM_50d_100_vocab.pkl",
        }


    """
        GRU feature의 경우 미리 pre-train된 GloVe 벡터를 불러와 이용하는 방식이므로 fit이 필요하지 않습니다.
    """
    def transform(self, X):
        """
            pretrain된 GloVe vector를 이용해 GRU input sequence로 변환해주는 메소드

        :param X: [[head sentence, body sentence], [head2, body2], ..., [headN, bodyN]] 형식의 2차원 리스트

        :return: input된 기사(head+body)의 100단어 sequence index 리스트들의 리스트

        [[sent0_idx_0, sent0_idx_1, sent0_idx_2, ..., sent0_idx_99],
        [...],
        [sentN_idx_0, sentN_idx_1, sentN_idx_2, ..., sentN_idx_99]]

        output shape : (n, 100)
        """
        print('Load GloVe embedding file...')
        GloVe_vectors = load_embedding_pandas(self.param_dict["GLOVE_ZIP_FILE"], self.param_dict["GLOVE_FILE"])
        print('Done.')
        
        data = [x[0]+". "+x[1] for x in X]


        vocab = create_embedding_lookup_pandas(data, self.param_dict["MAX_NB_WORDS"], self.param_dict["EMBEDDING_DIM"],
                                               GloVe_vectors, self.param_dict["EMBEDDING_FILE"], self.param_dict["VOCAB_FILE"],
                                               init_zeros=False, add_unknown=True, rdm_emb_init=True, tokenizer=nltk.word_tokenize)
        del GloVe_vectors

        sequences = text_to_sequences_fixed_size(data, vocab, self.param_dict["MAX_SEQ_LENGTH"],
                                                 save_full_text=False, take_full_claim=True)
        return sequences

if __name__ == "__main__":
    test_X = [["hello", "hello my name is lee"],
              ["bye", "bye my name is lee."],
              ["car", "bus car train"],
              ["train", "apple, train, bus"],
              ["bow", "sword, news, bow"],
              ["subway", "news, boy, line"],
              ["pot", "cow, soccer"],
              ["row", "low, huge, big, large"],
              ["much", "bird"],
              ["poor", "quite"]]

    gru_vector = GRUVectorizer()

    seq = gru_vector.transform(test_X)
    print(seq)