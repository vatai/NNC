"""
Examine the different layers in the keras_applications pretrained
networks.

The result is a dictionary, with layer name as key, and a list of
model names which contain the given layer.
"""

from pickle import dump
from nnclib.model_dict import model_dict

if __name__ == '__main__':
    stats = {}
    for name, model_cls in model_dict.items():
        model_cls = model_cls[0]
        model = model_cls()
        for layer in model.layers:
            ltype = type(layer)
            if ltype in stats:
                stats[ltype].append(name)
            else:
                stats[ltype] = [name]
        print("{} done.".format(name))

        dump(stats, open("layer_stats.pickle", 'bw'))
