from random import randint

from utilities import Utilities
from lav import LanguageAndVision

lav, utilities = LanguageAndVision(), Utilities()

all_items_in_train, all_items_in_val, all_items_in_test = utilities.load_data()  # load features
query_item = all_items_in_test[randint(0, len(all_items_in_test))]  # choose a random query
candidate_items = all_items_in_train + all_items_in_val  # define reference set

descriptions = lav.describe_image(query_item, candidate_items[0:])  # describe image
print "%s\t%s" % (descriptions[0][0], query_item[3])  # display query image and its caption
