import numpy as np
from vision import Vision
from language import Language
from utilities import Utilities


class LanguageAndVision:

    def __init__(self):
        pass

    @staticmethod
    def describe_image(query_item, candidate_items):
        """ Retrieve relevant descriptions for a query image """

        vision = Vision()
        language = Language()
        utilities = Utilities()

        settings = utilities.load_settings('settings.ini', 'Settings')
        neighbours = vision.retrieve_visually_similar_images(query_item, candidate_items, int(settings['farneighborsize']), settings)

        # ---------------------------------VISUALLY CLOSEST--------------------------------
        visually_closest_captions = []
        for vcCaption in neighbours[0][int(settings["captionindex"])]:  # find visually closest candidates' captions
            visually_closest_captions.append(vcCaption)

        neighbours, dist_min, dist_max = vision.remove_outliers(neighbours, int(settings["vdsindex"]),
                                                                float(settings["epsilon"]), int(settings["neighborsize"]))

        # ---------------------------- START MAIN PROCEDURE ----------------------------------

        w, model, vocab = language.load_word_models(settings)
        word_list_to_exclude = language.load_word_list_to_exclude(settings)
        sample_word_vector = language.get_word_vector('a', settings['method'], model, w, vocab)
        zeros = np.zeros(sample_word_vector.shape)

        number_of_tokens = []
        if settings['usesentencelengthpenalty'] == '1':
            for i, neighbour in enumerate(neighbours):
                for j, caption in enumerate(neighbour[int(settings["captionindex"])]):
                    if len(caption) == 1:
                        tokens = caption.split()  # tokenize by whitespace
                    else:
                        tokens = caption  # sentence is already tokenized
                    token_count = len(tokens)
                    number_of_tokens.append(token_count)
            average_token_count = int(np.mean(number_of_tokens))

        # --------------------COMPUTE QUERY VECTOR------------------------------
        total_vector_sum = zeros
        all_caption_vector_items = []
        oov_count = 0
        token_count = 0

        for i, neighbour in enumerate(neighbours):  # for each candidate
            caption_vector_sum = zeros  # to store caption vectors

            if settings['usevisualsimilarityscores'] == '1':
                visual_similarity_score = vision.compute_visual_similarity(neighbour[int(settings["vdsindex"])], dist_min, dist_max)
            else:
                visual_similarity_score = 1  # no effect

            for j, caption in enumerate(neighbour[int(settings["captionindex"])]):
                caption_vector, number_of_tokens, number_of_oovs = language.compute_sentence_vector(caption, model, w, vocab, zeros,
                                                                                                    word_list_to_exclude, settings)
                token_count = token_count + number_of_tokens
                oov_count = oov_count + number_of_oovs
                index_of_item = i + 1  # the index of item in the original list
                caption_vector = caption_vector * visual_similarity_score  # weighted summation with visual distance

                if settings['usesentencelengthpenalty'] == '1':
                    penalty = language.compute_sentence_length_penalty(number_of_tokens, average_token_count)
                    caption_vector = caption_vector * penalty

                caption_vector_item = [index_of_item, caption_vector, caption]
                all_caption_vector_items.append(caption_vector_item)
                caption_vector_sum += caption_vector

            total_vector_sum = total_vector_sum + caption_vector_sum

        query_vector = np.divide(total_vector_sum, len(all_caption_vector_items))

        cosine_similarities = []

        for caption_vector_item in all_caption_vector_items:
            cosine_similarity = language.compute_cosine_similarity(query_vector, caption_vector_item[1])  # 2nd index holds caption vector
            cosine_similarities.append(cosine_similarity)
            caption_vector_item.append(cosine_similarity)

        all_caption_vector_items.sort(key=lambda x: x[3], reverse=True)  # sort by 4th column, that is cosine similarity

        candidate_translations = []  # select top N descriptions from the results
        for i, caption_vector_item in enumerate(all_caption_vector_items[0:int(settings['numberofcaptionstoreturn'])]):
            candidate_translations.append(caption_vector_item[2])

        reference_translations = []
        for i, query_caption in enumerate(query_item[int(settings["captionindex"])]):
            reference_translations.append(query_caption)

        oov_rate = oov_count * 100 / token_count

        return [candidate_translations, reference_translations, visually_closest_captions, oov_rate]
