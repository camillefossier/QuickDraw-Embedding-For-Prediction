import ndjson

import numpy as np
import sklearn.linear_model as sklm
import sklearn.mixture as skmx
import esig.tosig as ts

from drawing import Drawing
from sklearn import metrics
from sklearn.cluster import SpectralClustering
import sklearn as sk

from timeit import default_timer as timer
from numba import jit

import cProfile

class Evaluator:

    CLASSIFICATION = 0
    CLUSTERING = 1

    VECTOR = 0
    MATRIX = 1
    MATRIX_LIST = 2

    NOT_IMPLEMENTED_MESSAGE = "Evaluator is an abstract class."
    UNKNOWN_SHAPE_MESSAGE = "Unknown input shape."

    def __init__(self, shape):
        self.task = None
        self.shape = abs(int(shape))
        self.model = None
        if self.shape > 2:
            raise Exception("Incorrect shape value")
    
    def fit(self, X, y=None):
        raise NotImplementedError(Evaluator.NOT_IMPLEMENTED_MESSAGE)

    def predict(self, X):
        raise NotImplementedError(Evaluator.NOT_IMPLEMENTED_MESSAGE)
    
    def accuracy(self, X, y):
        raise NotImplementedError(Evaluator.NOT_IMPLEMENTED_MESSAGE)

    def reshape(self, data, input_shape):
        if self.shape is Evaluator.VECTOR:
            if input_shape is Embedding.VECTOR_SHAPE:
                return np.array(data)
            elif input_shape is Embedding.MATRIX_SHAPE:
                return np.array([draw.flatten() for draw in data])
            elif input_shape is Embedding.MATRIX_LIST_SHAPE:
                return np.array([np.array([stroke.flatten() for stroke in draw]) for draw in data])
            else:
                raise Exception(Evaluator.UNKNOWN_SHAPE_MESSAGE)
        else:
            raise NotImplementedError("Evaluator reshape method not yet implemented for this shape.")

class ClassifierEvaluator(Evaluator):
    def __init__(self, shape):
        super().__init__(shape)
        self.task = Evaluator.CLASSIFICATION

    def accuracy(self, X, y):
        score = self.predict(X) == y
        return np.sum(score) / score.shape[0]

class LogisticRegressorEvaluator(ClassifierEvaluator):
    def __init__(self, multi_class="multinomial",solver="lbfgs", C=10, max_iter=7000):
        super().__init__(Evaluator.VECTOR)
        self.model = sklm.LogisticRegression(multi_class=multi_class, solver=solver, C=C, max_iter=max_iter)
        #if self.shape != VECTOR:
        #    raise Exception("Logistic Regression only accepts 1-dimensional vectors.")

    def fit(self, X, y=None):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)

class SVMEvaluator(ClassifierEvaluator):
    def __init__(self, multi_class="multinomial",solver="lbfgs", C=10, max_iter=7000):
        super().__init__(Evaluator.VECTOR)
        self.model = sk.svm.SVC()
        #if self.shape != VECTOR:
        #    raise Exception("Logistic Regression only accepts 1-dimensional vectors.")

    def fit(self, X, y=None):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)

class CNNEvaluator(ClassifierEvaluator):
    def __init__(self, multi_class="multinomial",solver="lbfgs", C=10, max_iter=7000):
        super().__init__(Evaluator.VECTOR)
        self.model = sk.svm.SVC()
        # TODO
        pass
        
    def fit(self, X, y=None):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)

class ClusteringEvaluator(Evaluator):
    def __init__(self, shape, nb_clusters):
        super().__init__(shape)
        self.task = Evaluator.CLUSTERING
        self.nb_clusters = nb_clusters

    def accuracy(self, X, y):
        prediction = self.predict(X)
        return metrics.v_measure_score(y, prediction)

class SpectralClusteringEvaluator(ClusteringEvaluator):
    def __init__(self, nb_clusters):
        super().__init__(Evaluator.VECTOR, nb_clusters)
        self.model = SpectralClustering(nb_clusters, n_init=100, assign_labels='discretize')
        self.prediction = None

    def fit(self, X, y=None):
        self.prediction = self.model.fit_predict(X)

    def predict(self, X):
        return self.prediction

class EMClusteringEvaluator(ClusteringEvaluator):
    def __init__(self, nb_clusters):
        super().__init__(Evaluator.VECTOR, nb_clusters)
        self.model = skmx.GaussianMixture(n_components=nb_clusters, covariance_type='spherical')

    def fit(self, X, y=None):
        self.model.fit_predict(X)

    def predict(self, X):
        return self.model.predict(X)
  
class Embedding:

    NOT_IMPLEMENTED_MESSAGE = "Embedding is an abstract class."

    UNDEFINED_SHAPE = -1
    VECTOR_SHAPE = 0
    MATRIX_SHAPE = 1
    MATRIX_LIST_SHAPE = 2

    def __init__(self):
        self.output_shape = Embedding.UNDEFINED_SHAPE
    
    def embed(self, data):
        raise NotImplementedError(Embedding.NOT_IMPLEMENTED_MESSAGE)

    def post_embedding(self, data):
        return data
    
    def shape(self, data):
        return self.output_shape

class Signature(Embedding):
    def __init__(self, degree, log=False):
        super()
        self.output_shape = Embedding.VECTOR_SHAPE
        self.degree = degree
        self.log = log

    def embed(self, data):
        return ts.stream2sig(data.concat_drawing(), self.degree) if not self.log else ts.stream2logsig(data.concat_drawing(), self.degree)

class TDA(Embedding):
    def __init__(self, abscissa=Drawing.T, ordinate=Drawing.Y, width=1, spacing=1, offset=1, concat=True, clipping=False):
        super()
        # TODO : Add possibility for list of matrices
        self.output_shape = Embedding.MATRIX_SHAPE
        self.abscissa = abscissa
        self.ordinate = ordinate
        self.width = width
        self.spacing = spacing
        self.offset = offset
        self.concat = concat
        self.clipping = clipping
    
    # TODO : Needs to sample only part of the points
    def embed(self, data):
        fen = 2 * self.width + 1 # Taille de la fenetre (nb colonnes dans la mat)
        final = [] # Matrice finale avec toutes les strokes concaténées

        if self.concat: # On souhaite avoir la version concaténée ? 
            strokes = [data.concat_drawing()]
        else:
            strokes = data.strokes
        for stroke in strokes:
            stroke = stroke[stroke[:,self.abscissa].argsort()] # Tri selon la variable abscisse
            shape = stroke.shape[0] # nb lignes
            mat = np.zeros((shape, fen)) # Initialisation de la matrice pour une stroke
            le = self.width * self.spacing 
            for i,depart in enumerate(range(le, shape - le, self.offset)):
                mat[i] = stroke[range(depart - le, depart + le + 1, self.spacing), self.ordinate]
            final.append(mat[0:i+1])
        return np.concatenate(final, axis = 0)
    
    # TODO
    def post_embedding(self, data):
        if self.clipping:
            min_width = min([d.shape[0] for d in data])


class Config:
    def __init__(self, embedding, evaluator):
        # Data ==> Embedding ==> Reshape ==> Evaluate
        self.embedding = embedding
        self.evaluator = evaluator

class Tester:
    def __init__(self, 
                datasets_path,
                configs,
                store_data=True,
                nb_lines=None,
                train_ratio=0.8,
                do_link_strokes=False,
                do_rescale=False,
                link_steps=None,
                link_timestep=None):
        self.datasets_path = datasets_path
        self.configs = configs
        self.store_data = store_data
        self.nb_lines = nb_lines
        self.train_ratio = 0.8
        self.do_link_strokes = do_link_strokes
        self.do_rescale = do_rescale
        self.link_steps = link_steps
        self.link_timestep = link_timestep
        self.stored_data = {}
        self.init_data_storage()
    
    def init_data_storage(self):
        for path in self.datasets_path:
            if path not in self.stored_data:
                self.stored_data[path] = {"data":None}
            for config in self.configs:
                if config.embedding not in self.stored_data[path]:
                    self.stored_data[path][config.embedding] = None

    def run(self):
        for config in self.configs:
            train_data = []
            train_labels = []
            for path in self.datasets_path:
                # Extract data
                if self.stored_data[path][config.embedding] is None:    
                    if self.stored_data[path]["data"] is None:
                        data = self.read_data(path)
                        if self.store_data:
                            self.stored_data[path]["data"] = data
                    else:
                        data = self.stored_data[path]["data"]
                    embedding = [config.embedding.embed(draw) for draw in data]
                    embedding = config.embedding.post_embedding(embedding)
                    if self.store_data:
                        self.stored_data[path][config.embedding] = embedding
                else:
                    data = self.stored_data[path]["data"]
                    embedding = self.stored_data[path][config.embedding]
                train_labels.append([d.label for d in data])
                train_data = train_data + embedding
            
            X = config.evaluator.reshape(train_data, config.embedding.output_shape)
            y = np.concatenate(train_labels)

            if config.evaluator.task == Evaluator.CLASSIFICATION:
                n = X.shape[0]
                lim = int(self.train_ratio * n)
                shuffle = np.arange(0, n)
                np.random.shuffle(shuffle)

                X_train = X[shuffle][:lim]
                y_train = y[shuffle][:lim]
                X_test = X[shuffle][lim:]
                y_test = y[shuffle][lim:]

                config.evaluator.fit(X_train, y=y_train)
                print(config.evaluator.accuracy(X_test, y_test))
            
            elif config.evaluator.task == Evaluator.CLUSTERING:
                config.evaluator.fit(X)
                print(config.evaluator.accuracy(X, y))

    def read_data(self, path):
        if self.nb_lines is not None:
            lines = self.nb_lines * [None]
            with open(path) as f:
                for i in range(self.nb_lines):
                    line = f.readline()
                    if line:
                        lines[i] = line
                    else:
                        break
            data = ndjson.loads("".join(lines))
        else:
            with open(path) as f:
                data = ndjson.load(f)
        
        return [Drawing(draw,
                        do_link_strokes=self.do_link_strokes,
                        do_rescale=self.do_rescale,
                        link_steps=self.link_steps,
                        link_timestep=self.link_timestep) for draw in data]

if __name__ == '__main__':
    lr = LogisticRegressorEvaluator()
    svm = SVMEvaluator()
    signature = Signature(4, log=True)
    tda = TDA()
    
    config = Config(signature, lr)
    config2 = Config(tda, lr)
    config3 = Config(signature, SpectralClusteringEvaluator(5))
    config4 = Config(signature, EMClusteringEvaluator(5))
    config5 = Config(signature, svm)

    tester = Tester(
        ["../data/full_raw_axe.ndjson", "../data/full_raw_sword.ndjson", "../data/full_raw_squirrel.ndjson",
        "../data/full_raw_The Eiffel Tower.ndjson", "../data/full_raw_basketball.ndjson"],
        [config, config5],
        store_data=True,
        nb_lines=1000,
        do_link_strokes=True,
        do_rescale=True,
        link_steps=2,
    )
    #cProfile.run('tester.run()', sort="cumtime")
    start = timer()
    tester.run()
    print(timer() - start)
    
    exit(0)

    import sklearn.decomposition
    pca = sklearn.decomposition.PCA()
    trans = pca.fit(X)

    import matplotlib.pyplot as plt

    colors = np.copy(y)
    colors[colors == 'axe'] = 0
    colors[colors == 'sword'] = 1
    colors[colors == 'squirrel'] = 2
    colors[colors == 'basketball'] = 3
    colors[colors == 'The Eiffel Tower'] = 4
    colors = colors.astype(float)

    plt.scatter(trans[:,0], trans[:,1], c=colors, s=2)

    plt.show()

# TODO :
# PCA for clustering ? Non pas la priorité
# Visualization for clusters ? mouais
# Storing results ! oui
# Add parameters to evaluators