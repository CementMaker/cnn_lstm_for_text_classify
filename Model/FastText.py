import numpy as np

from fastText import train_supervised
from sklearn.metrics import classification_report


def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))


if __name__ == "__main__":
    train_data = '../data/fastTextData/train_data.txt'
    valid_data = '../data/fastTextData/valid_data.txt'

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

    predict_label = []
    for line in test_data:
        result = model.predict(line)
        predict_label.append(result[0][0][-2])
    print(classification_report(y_pred=predict_label, y_true=test_label))
