import ndjson
import numpy as np

from drawing import Drawing
from datetime import datetime
from sklearn.linear_model import LogisticRegression

# https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/raw

NB_LINES = 5000

if __name__ == '__main__':
        
    objects = ["./data/full_raw_axe.ndjson",
               "./data/full_raw_sword.ndjson",
               "./data/full_raw_squirrel.ndjson"]
    X = None
    y = None
    for obj in objects:
        start = datetime.now()
        with open(obj) as f:
            lines = "".join(f.readlines()[:NB_LINES])
        data = ndjson.loads(lines)
        print(datetime.now() - start)
        
        start = datetime.now()
        drawings = [Drawing(draw, do_link_strokes=True, do_rescale=True, link_steps=2) for draw in data]
        print(datetime.now() - start)
        
        start = datetime.now()
        signatures = np.array([draw.signature(4, log=False) for draw in drawings])
        print(datetime.now() - start)

        if X is None:
            X = signatures
        else:
            X = np.r_[X, signatures]
        if y is None:
            y = np.repeat(drawings[0].label, NB_LINES, axis=0)
        else:
            y = np.r_[y, np.repeat(drawings[0].label, NB_LINES, axis=0)]

    n = X.shape[0]
    lim = int(0.8 * n)
    shuffle = np.arange(0, n)
    np.random.shuffle(shuffle)

    X_train = X[shuffle,:][:lim,:]
    y_train = y[shuffle][:lim]
    X_test = X[shuffle,:][lim:,:]
    y_test = y[shuffle][lim:]

    softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10, max_iter=7000)
    softmax_reg.fit(X_train, y_train)

    accuracy = np.sum(softmax_reg.predict(X_test) == y_test) / y_test.shape[0]

    print("ok")