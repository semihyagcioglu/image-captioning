import numpy as np
import ConfigParser


class Utilities:

    def __init__(self):
        pass

    @staticmethod
    def load_settings(file_name, section_name):
        config = ConfigParser.ConfigParser()
        config.read(file_name)
        settings = dict(config.items(section_name))  # access as settings['method']
        return settings

    def load_data(self):

        settings = self.load_settings('settings.ini', 'Settings')

        # ------------------- LOAD DATA -----------------------
        annotationPathForTest = settings['datapath'] + 'mRNN/vgg-original-mscoco-2014/mscoco_anno_files/anno_list_mscoco_test2014.npy'
        testSplitAnnotations = np.load(annotationPathForTest)
        annotationPathForTrain = settings[
                                     'datapath'] + 'mRNN/vgg-original-mscoco-2014/mscoco_anno_files/anno_list_mscoco_trainModelVal_m_RNN.npy'
        trainSplitAnnotations = np.load(annotationPathForTrain)
        annotationPathForVal = settings['datapath'] + 'mRNN/vgg-original-mscoco-2014/mscoco_anno_files/anno_list_mscoco_crVal_m_RNN.npy'
        valSplitAnnotations = np.load(annotationPathForVal)
        vggFilePath = settings['datapath'] + 'mRNN/vgg-original-mscoco-2014/image_features_mRNN/VGG_feat_o_dct_mscoco_2014.npy'
        vgg = np.load(vggFilePath).tolist()
        all_items = []
        for item in trainSplitAnnotations:
            all_items.append([])
            i = len(all_items) - 1
            all_items[i].append(item["id"])
            all_items[i].append(item["file_path"])
            all_items[i].append(item["file_name"])
            all_items[i].append(item["url"])
            all_items[i].append(str('train'))
            all_items[i].append(vgg[item["id"]])  # add feature
            sentences = []
            for sentence in item["sentences"]:
                sentence = ' '.join([str(item).strip() for item in sentence])
                sentences.append(sentence)
            all_items[i].append(sentences)
        for item in valSplitAnnotations:
            all_items.append([])
            i = len(all_items) - 1
            all_items[i].append(item["id"])
            all_items[i].append(item["file_path"])
            all_items[i].append(item["file_name"])
            all_items[i].append(item["url"])
            all_items[i].append(str('val'))
            all_items[i].append(vgg[item["id"]])  # add feature
            sentences = []
            for sentence in item["sentences"]:
                sentence = ' '.join([str(item).strip() for item in sentence])
                sentences.append(sentence)
            all_items[i].append(sentences)
        for item in testSplitAnnotations:
            all_items.append([])
            i = len(all_items) - 1
            all_items[i].append(item["id"])
            all_items[i].append(item["file_path"])
            all_items[i].append(item["file_name"])
            all_items[i].append('http://tasviret.cs.hacettepe.edu.tr/dataset/MSCOCO/test2014/' + item["file_name"])
            all_items[i].append(str('test'))
            all_items[i].append(vgg[item["id"]])  # add feature
            sentences = []
            for step in range(1, 6, 1):
                sentences.append('')
            all_items[i].append(sentences)

        return all_items
