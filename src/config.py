from sklearn.linear_model import LogisticRegression

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

    def predict(X)
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)
    
    def accuracy(X, y)
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

class ClassifierEvaluator(Evaluator):

    def __init__(self, shape):
        super(shape)

    def accuracy(X, y):
        score = self.predict(X) == y
        return np.sum(score) / score.shape[0]

class LogisticRegressorEvaluator(ClassifierEvaluator):

    def __init__(self, shape, multi_class="multinomial",solver="lbfgs", C=10, max_iter=7000):
        super(shape)
        if self.shape != VECTOR:
            raise Exception("Logistic Regression only accepts 1-dimensional vectors.")
        self.model = LogisticRegression(multi_class=multi_class, solver=solver, C=C, max_iter=max_iter)
    
    def fit(X, y=None):
        self.model.fit(X, y)
    
    def predict(X):
        return self.model.predict(X)

class Config:
    def __init__(self, data, evaluators):
        self.data = {}
        self.evaluators = evaluators

