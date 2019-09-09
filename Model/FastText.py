import numpy as np

from tqdm import tqdm
from fastText import train_supervised
from sklearn.metrics import classification_report


'''
参考链接
    https://github.com/facebookresearch/fastText
    https://github.com/facebookresearch/fastText/blob/master/python/doc/examples/train_supervised.py
'''


def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))


if __name__ == "__main__":
    train_data = '../data/fastTextData/train_data'
    valid_data = '../data/fastTextData/valid_data'

    test_data = [line.strip().split('\t')[1] for line in open(valid_data, "r")]
    test_label = [line.strip().split('\t')[0] for line in open(valid_data, "r")]
    model = train_supervised(input=train_data,
                             dim=100,
                             lr=0.1,
                             wordNgrams=2,
                             minCount=1,
                             bucket=10000000,
                             epoch=6,
                             thread=4,
                             label='__label__')

    print(np.array(test_data).shape)
    print(np.array(test_label).shape)
    print(model.test("../data/fastTextData/valid_data"))

    predict_label = []
    for line in tqdm(test_data):
        result, proba = model.predict(line)
        predict_label.append(result[0])
    print(classification_report(y_pred=predict_label, y_true=test_label))
