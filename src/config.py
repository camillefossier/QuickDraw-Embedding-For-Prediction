from sklearn.linear_model import LogisticRegression
import numpy as np

class Evaluator:

    VECTOR = 0
    MATRIX = 1
    MATRIX_LIST = 2

    NOT_IMPLEMENTED_MESSAGE = "Evaluator is an abstract class."

    def __init__(self, shape):
        self.shape = abs(int(shape))
        self.model = None
        if self.shape > 2:
            raise Exception("Incorrect shape value")
    
    def fit(X, y=None):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def predict(X):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)
    
    def accuracy(X, y):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)
    
    def reshape():
        0

class ClassifierEvaluator(Evaluator):
    def __init__(self, shape):
        super(shape)

    def accuracy(X, y):
        score = self.predict(X) == y
        return np.sum(score) / score.shape[0]

class LogisticRegressorEvaluator(ClassifierEvaluator):
    def __init__(self, shape, multi_class="multinomial",solver="lbfgs", C=10, max_iter=7000):
        super(shape)
        self.model = LogisticRegression(multi_class=multi_class, solver=solver, C=C, max_iter=max_iter)
        #if self.shape != VECTOR:
        #    raise Exception("Logistic Regression only accepts 1-dimensional vectors.")

    def fit(X, y=None):
        self.model.fit(X, y)
    
    def predict(X):
        return self.model.predict(X)

class Embedding:

    NOT_IMPLEMENTED_MESSAGE = "Embedding is an abstract class."

    UNDEFINED_SHAPE = -1
    VECTOR_SHAPE = 0
    MATRIX_SHAPE = 1
    MATRIX_LIST_SHAPE = 2

    def __init__(self):
        self.output_shape = UNDEFINED_SHAPE
    
    def embed(self, data):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)
    
    def output_shape(self, data):
        return self.output_shape

class Signature(Embedding):
    def __init__(self, degree):
        super()
        self.output_shape = VECTOR_SHAPE
        self.degree = degree

    def embed(self, data):
        # TODO : Return signature

class TDA(Embedding):
    def __init__(self, width, spacing, offset):
        super()
        self.width = width
        self.spacing = spacing
        self.offset = offset
    
    def embed(self, data):
        # TODO : Return TDA

class Config:
    def __init__(self, embedding, evaluator):
        self.embedding = embedding
        self.evaluator = evaluator

        # Data ==> Embedding ==> Reshape ==> Evaluate

class Tester:
    def __init__(self, datasets_path, configs, store_data=True, nb_lines=None, train_ratio=0.8):
        self.datasets_path = datasets_path
        self.configs = configs
        self.store_data = store_data
        self.nb_lines = nb_lines
        self.train_ratio = 0.8
        self.stored_data = {}
        self.init_data_storage()
    
    def init_data_storage(self):
        for path in self.datasets_path:
            if path not in self.stored_data:
                self.stored_data[path] = {"data":None}
            for config in configs:
                if config.embedding not in self.stored_data[path]:
                    self.stored_data[path][config.embedding] = None

    def run(self):
        for config in self.configs:
            train_data = []
            train_labels = []
            for path in self.datasets_path:
                # Extract data
                if self.stored_data[path][config] is None:    
                    if stored_data[path] is None:
                        data = self.read_data(path)
                        if self.store_data:
                            stored_data[path]["data"] = data
                    else:
                        data = self.stored_data[path]["data"]
                    train_labels.append([d.label for d in data])
                    embedding = config.embedding.embed(data)
                    if self.store_data:
                        self.stored_data[path][config.embedding] = embedding
                else:
                    embedding = self.stored_data[path][config.embedding]
                train_data.append(embedding)
            
            X = config.evaluator.reshape(train_data)
            y = np.concatenate(train_labels)

            n = X.shape[0]
            lim = int(0.8 * n)
            shuffle = np.arange(0, n)
            np.random.shuffle(shuffle)

            X_train = X[shuffle,:][:lim,:]
            y_train = y[shuffle][:lim]
            X_test = X[shuffle,:][lim:,:]
            y_test = y[shuffle][lim:]
            
            config.evaluator.fit(X_train, y=y_train)
            print(config.evaluator.accuracy(X_test, y_test))
    
    def read_data(self, path):
        if self.nb_lines is not None:
            with open(path) as f:
                lines = "".join(f.readlines()[:self.nb_lines])
            data = ndjson.loads(lines)
        else:
            with open(path) as f:
                data = ndjson.load(f)
        
        # TODO : Parameters
        return [Drawing(draw, do_link_strokes=True, do_rescale=True, link_steps=2) for draw in data]