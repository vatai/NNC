from time import time

import nnclib.experiments.collection as collection

print('>>>>>>>>>> VERY BEGINNING <<<<<<<<<<', time())
collection.inceptionresnetv2_experiment.run()
