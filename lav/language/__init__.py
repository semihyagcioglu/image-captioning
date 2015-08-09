from __future__ import division
import numpy as np
import lib.word2vec.word2vec as word2vec  # for word2vec


class Language:

    def __init__(self):
        pass

    @staticmethod
    def load_word_models(settings):
        if settings['method'] == 'word2vec':
            model_file_for_word2vec = settings['filepath'] + settings['corpuspath'] + settings['word2vecvectorfilename']
            model = word2vec.load(model_file_for_word2vec)
            w = None  # for glove compatibility
            vocab = None  # for glove compatibility

        return w, model, vocab

    @staticmethod
    def load_word_list_to_exclude(settings):
        """Load the list of words that should be excluded based on the algorithm parameters"""

        stop_words = list()
        operator_words = list(['and', 'or', 'not'])

        if settings['excludestopwords'] == '1':
            stop_words = list(
                ['all', 'just', 'being', 'over', 'both', 'through', 'yourselves', 'its', 'before', 'herself', 'had', 'should',
                 'to', 'only', 'under', 'ours', 'has', 'do', 'them', 'his', 'very', 'they', 'not', 'during', 'now', 'him',
                 'nor', 'did', 'this', 'she', 'each', 'further', 'where', 'few', 'because', 'doing', 'some', 'are', 'our',
                 'ourselves', 'out', 'what', 'for', 'while', 'does', 'above', 'between', 't', 'be', 'we', 'who', 'were', 'here',
                 'hers', 'by', 'on', 'about', 'of', 'against', 's', 'or', 'own', 'into', 'yourself', 'down', 'your', 'from',
                 'her', 'their', 'there', 'been', 'whom', 'too', 'themselves', 'was', 'until', 'more', 'himself', 'that', 'but',
                 'don', 'with', 'than', 'those', 'he', 'me', 'myself', 'these', 'up', 'will', 'below', 'can', 'theirs', 'my',
                 'and', 'then', 'is', 'am', 'it', 'an', 'as', 'itself', 'at', 'have', 'in', 'any', 'if', 'again', 'no', 'when',
                 'same', 'how', 'other', 'which', 'you', 'after', 'most', 'such', 'why', 'a', 'off', 'i', 'yours', 'so', 'the',
                 'having', 'once'])

        word_list_to_exclude = stop_words
        word_list_to_exclude = set(word_list_to_exclude)

        if (settings['excludestopwords'] == '1') and settings['includeoperatorwords'] == '1':
            word_list_to_exclude = word_list_to_exclude - set(operator_words)

        return word_list_to_exclude

    @staticmethod
    def get_word_vector(word, method, model, w, vocab):

        word_vector = 0
        if method == 'word2vec':
            word_vector = model[word]

        return word_vector

    def compute_sentence_vector(self, sentence, model, w, vocab, zeros, word_list_to_exclude, settings):

        if len(sentence) == 1:
            tokens = sentence.split()  # tokenize by whitespace
        else:
            tokens = sentence  # sentence is already tokenized

        sentence_vector = np.longdouble(zeros)
        oov_count = 0

        for token in tokens:
            token = token.strip()

            try:
                if (settings['excludestopwords'] == '1') and token in word_list_to_exclude:
                    continue  # token somehow is not eligible to be used in our representation

                token_vector = self.get_word_vector(token, settings['method'], model, w, vocab)

            except Exception, e:
                # print(str(e))
                oov_count += 1
            else:
                token_vector = np.longdouble(np.asarray(token_vector))
                sentence_vector = sentence_vector + token_vector  # make vector summation for each token

        token_count = len(tokens)
        return sentence_vector, token_count, oov_count

    @staticmethod
    def compute_sentence_length_penalty(token_count, average_token_count):
        """ Compute the penalty score using the length of the candidate sentence
        according to the average token count."""

        penalty = 1 - (abs(token_count - average_token_count) / max(token_count, average_token_count))
        return penalty

    @staticmethod
    def compute_cosine_similarity(base_vector, target_vector):
        """Compute the cosine similarity between two vectors based on the angular cosine distance
        return range -1 to 1, where 1 means two vectors are identical,
        -1 means reverse!*!, 0 means vectors are orthogonal
        where cosine(A,B) = dot(A,B) / ( || A || * || B || ) """

        np.seterr(all='print')
        cosine_similarity = 0

        try:
            base_vector = np.longdouble(base_vector)
            target_vector = np.longdouble(target_vector)
            vector_dot_products = np.dot(base_vector, target_vector)
            vector_norms = np.linalg.norm(base_vector) * np.linalg.norm(target_vector)
            cosine_similarity = np.divide(vector_dot_products, vector_norms)

            if vector_norms == 0.0:
                print 'Error in vec in compute_cosine_similarity'
                print target_vector

        except Exception, e:
            print(str(e))

        return cosine_similarity
