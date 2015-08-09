from random import randint

from utilities import Utilities
from lav import LanguageAndVision

lav = LanguageAndVision()
utilities = Utilities()

all_items = utilities.load_data()

all_items_in_train = filter(lambda x: x[4] == 'train', all_items)
all_items_in_val = filter(lambda x: x[4] == 'val', all_items)
all_items_in_test = filter(lambda x: x[4] == 'test', all_items)
candidate_items = all_items_in_train + all_items_in_val
queryInd = randint(0, len(all_items_in_test))
query_item = all_items_in_test[queryInd]
query_item.append(0.0)  # Append zero as distance to make structs similar


descriptions = lav.describe_image(query_item, candidate_items[0:])

for i, candidateTranslation in enumerate(descriptions[0]):
    print "%s\t%s" % (candidateTranslation, query_item[3])
