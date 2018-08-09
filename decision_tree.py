import numpy as np
from sklearn.datasets import load_iris
from pprint import pprint

def splitfxn(features, label):
    if len(set(label)) == 1 or len(label) == 0:
        return label
    igs = np.apply_along_axis(information_gain, axis=0, arr=features, label=label)
    best_feature = features[:, np.argmax(igs[0, :])]
    best_alpha = igs[1, np.argmax(igs[0, :])]

    # If there's no gain at all, nothing has to be done, just return the original set
    if np.max(igs < 1e-10):
        return y

    if not np.isnan(best_alpha):
        tmp = np.ndarray(shape=best_feature.shape, dtype='U16')
        tmp[best_feature > best_alpha] = "> {}".format(best_alpha)
        tmp[best_feature <= best_alpha] = "<= {}".format(best_alpha)

    # We split using the selected attribute
    sets = partition(tmp)

    res = {}
    for k, v in sets.items():
        y_subset = label.take(v, axis=0)
        x_subset = features.take(v, axis=0)

        res["x_%d = %s" % (np.argmax(igs[0, :]), k)] = splitfxn(x_subset, y_subset)

    return res


def partition(a):
    return {c: (a == c).nonzero()[0] for c in np.unique(a)}


def entropy(label):
    _, counts = np.unique(label, return_counts=True)
    return -np.sum([cnt / len(label) * np.log2(cnt / len(label)) for cnt in counts])


def conditional_entropy(feature, label, split='median'):
    # check if discrete feature
    if issubclass(feature.dtype.type, np.integer):
        if np.all(np.unique(feature) == np.array(range(np.max(feature) + 1))):
            classes = np.unique(feature)
            return np.sum(
                [np.sum(feature == cls) / len(feature) * entropy(label[feature == cls]) for cls in classes]), np.NaN
    # treat continuous predictors
    elif split == 'bestSplit':
        splits = np.unique(np.sort(feature))
        igs = [information_gain(feature > alpha, label) for alpha in splits]
        alpha = splits[np.argmax(igs)]
        return (np.sum(feature > alpha) / len(label) * entropy(label[feature > alpha]) +
               np.sum(feature <= alpha) / len(label) * entropy(label[feature <= alpha]), alpha)
    else:
        alpha = np.median(feature)
        return (np.sum(feature > alpha) / len(label) * entropy(label[feature > alpha]) +
               np.sum(feature <= alpha) / len(label) * entropy(label[feature <= alpha]), alpha)


def information_gain(feature, label, split='median'):
    entr, alpha = conditional_entropy(feature, label, split)
    return entropy(label) - entr, alpha


# Test against sklearn
x, y = load_iris(True)
tree = splitfxn(x, y)

pprint(tree)