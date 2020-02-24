import ndjson

import numpy as np
import sklearn.linear_model as sklm
import sklearn.mixture as skmx
import esig.tosig as ts
import sklearn as sk
import scipy.interpolate as interpolate

from drawing import Drawing
from sklearn import metrics
from sklearn.cluster import SpectralClustering

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.utils import to_categorical

from timeit import default_timer as timer
from numba import jit

import cProfile

import matplotlib.pyplot as plt

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
    
    def fit(self, X, y=None, X_val=None, y_val=None):
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
        elif self.shape is Evaluator.MATRIX:
            if input_shape is Embedding.VECTOR_SHAPE:
                raise NotImplementedError("Cannot reshape vector to matrix")
            elif input_shape is Embedding.MATRIX_SHAPE:
                return np.array(data)
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

    def fit(self, X, y=None, X_val=None, y_val=None):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)

class SVMEvaluator(ClassifierEvaluator):
    def __init__(self, multi_class="multinomial",solver="lbfgs", C=10, max_iter=7000):
        super().__init__(Evaluator.VECTOR)
        self.model = sk.svm.SVC()
        #if self.shape != VECTOR:
        #    raise Exception("Logistic Regression only accepts 1-dimensional vectors.")

    def fit(self, X, y=None, X_val=None, y_val=None):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)

class CNNEvaluator(ClassifierEvaluator):
    def __init__(self, input_shape, num_classes, batch_size=50, epochs=15, num_filters=10, filter_size=(3,3), stride=(1,1), pool_size=(2,2), dropout=0):
        super().__init__(Evaluator.MATRIX)
        self.input_shape = input_shape # Keras need a third dimension
        self.num_classes = num_classes # Nombre de classes
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.pool_size = pool_size
        self.dropout = dropout
        self.model = self.init_model()

    def init_model(self):
        model = Sequential()
        model.add(Conv2D(filters = self.num_filters, kernel_size = self.filter_size, strides=self.stride,
                        activation = 'relu',
                        input_shape = self.input_shape))
        model.add(MaxPooling2D(pool_size = self.pool_size))
        model.add(Dropout(self.dropout)) # A voir
        model.add(Flatten()) # Flatten layers allow you to change the shape of the data from a vector of 2d matrixes (or nd matrices really) into the correct format for a dense layer to interpret.
        model.add(Dense(self.num_classes, activation='softmax')) # Couche de classif
        
        model.compile(
            optimizer = 'adam',                #  Adam gradient-based optimizer (Possibilité de changer)
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )

        return model

    def fit(self, X, y=None, X_val=None, y_val=None):
        
        for i,label in enumerate(np.unique(y)):
            y[y == label] = i
            y_val[y_val == label] = i

        X = np.expand_dims(X, 3)
        X_val = np.expand_dims(X_val, 3)
        y = to_categorical(y)
        y_val = to_categorical(y_val)
        
        self.model.fit(X, y,
                        batch_size=self.batch_size,
                        epochs=self.epochs,
                        verbose = 1, # 1 = Progress Bar visible
                        validation_data=(X_val, y_val))
    
    def predict(self, X):
        X = np.expand_dims(X, 3)
        return self.model.predict(X)
    
    def accuracy(self, X, y):
        X = np.expand_dims(X, 3)
        y = to_categorical(y)
        return self.model.evaluate(x=X, y=y,
            batch_size=self.batch_size,
            verbose=1,
            )


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

    def fit(self, X, y=None, X_val=None, y_val=None):
        self.prediction = self.model.fit_predict(X)

    def predict(self, X):
        return self.prediction

class EMClusteringEvaluator(ClusteringEvaluator):
    def __init__(self, nb_clusters):
        super().__init__(Evaluator.VECTOR, nb_clusters)
        self.model = skmx.GaussianMixture(n_components=nb_clusters, covariance_type='diag')

    def fit(self, X, y=None, X_val=None, y_val=None):
        self.model.fit_predict(X)

    def predict(self, X):
        return self.model.predict(X)
  
class Embedding:

    INSTANCES_COUNTS = 0

    NOT_IMPLEMENTED_MESSAGE = "Embedding is an abstract class."

    UNDEFINED_SHAPE = -1
    VECTOR_SHAPE = 0
    MATRIX_SHAPE = 1
    MATRIX_LIST_SHAPE = 2

    def __init__(self, embedding_id=None):
        if embedding_id:
            self.embedding_id = embedding_id
        else:
            Embedding.INSTANCES_COUNTS += 1
            self.embedding_id = str(Embedding.INSTANCES_COUNTS)
        self.output_shape = Embedding.UNDEFINED_SHAPE

    def embed(self, data):
        raise NotImplementedError(Embedding.NOT_IMPLEMENTED_MESSAGE)

    def post_embedding(self, data):
        return data
    
    def shape(self, data):
        return self.output_shape

class Signature(Embedding):
    def __init__(self, degree, log=False, nb_to_keep=None, embedding_id=None):
        super().__init__(embedding_id=embedding_id)
        self.output_shape = Embedding.VECTOR_SHAPE
        self.degree = degree
        self.log = log
        self.nb_to_keep = nb_to_keep

    def embed(self, data):
        return ts.stream2sig(data.concat_drawing(), self.degree) if not self.log else ts.stream2logsig(data.concat_drawing(), self.degree)
    
    def post_embedding(self, data):
        if self.nb_to_keep > 0:
            return [sig[:self.nb_to_keep] for sig in data]
        else:
            return data

class Spline(Embedding):
    def __init__(self, abscissa=Drawing.T, ordinate=[Drawing.X], applicate=None, nb_knots=15, degree=3, output_shape=Embedding.VECTOR_SHAPE):
        super()
        self.abscissa = abscissa
        self.ordinate = ordinate
        if type(self.ordinate) is not list:
            self.ordinate = [self.ordinate]
        self.applicate = applicate
        self.nb_knots = nb_knots
        self.degree = degree
        self.output_shape = output_shape
    
    def generate_knots(self, x):
        # knots t must satisfy the Schoenberg-Whitney conditions, i.e., there must
        # be a subset of data points x[j] such that t[j] < x[j] < t[j+k+1], for j=0, 1,...,n-k-2.
        n = self.nb_knots + 2
        ts = np.linspace(0, len(x)-1, n, dtype=np.int16)
        if not np.all(np.diff(ts) >= 2):
            return None
        return x[ts][1:-1]

    def embed(self, data):
        stroke = data.concat_drawing()
        # TODO 
        if self.applicate:
            stroke = data.concat_without_doubles([self.abscissa, self.ordinate])
        else:
            stroke = data.concat_without_doubles([self.abscissa])
        res = []
        for ordinate in self.ordinate:
            x = stroke[:, self.abscissa]
            y = stroke[:, ordinate]
            if self.applicate: # Bivariate
                z = stroke[:, self.applicate]

                # Param to modify
                """ tx = np.linspace(min(x), max(x), self.nb_knots)[1:-1]
                ty = np.linspace(min(y), max(y), self.nb_knots)[1:-1] """
                # tt = np.linspace(min(t), max(t), self.nb_knots)[1:-1] 
                x.sort()
                y.sort()
                tx = self.generate_knots(x)
                ty = self.generate_knots(y)
                if tx is None or ty is None:
                    return None

                # TODO : Add which columns to keep
                res = interpolate.bisplrep(x, y, z, kx=self.degree, ky=self.degree, tx=tx, ty=ty, task=-1)

            else: # Univariate
                order = x[:].argsort()
                x = x[order]
                y = y[order]

                # This function do the representation B-Spline for a 1D curve
                #coef = interpolate.splrep(x, y, s=stroke.shape[0]/600, k=4) # s = degree of smooth, k = degree of spline fit (4 = cubic splines)
                tx = self.generate_knots(x)
                if tx is None:
                    return None

                # return : A tuple (t,c,k) containing the vector of knots, the B-spline coefficients, and the degree of the spline
                res = res + list(interpolate.splrep(x, y, k=self.degree, t=tx, task=-1)[:2]) # s = degree of smooth, k = degree of spline fit (4 = cubic splines)
        res = np.array(res)
        if self.output_shape is Embedding.VECTOR_SHAPE:
            res = res.flatten()
        return res
        """
            spline = interpolate.BSpline(coef[0], coef[1], coef[2], extrapolate=False) # Do I return this ? 
            N = 100 
            xmin, xmax = x.min(), x.max()
            xx = np.linspace(xmin, xmax, N)
            plt.plot(x, y, 'bo', label='Original points')
            plt.plot(xx, spline(xx), 'r', label='BSpline')
            plt.grid()
            plt.legend(loc='best')
            plt.show()
        """

class TDA(Embedding):

    INTERPOLATION_THRESHOLD = 1

    def __init__(self, abscissa=Drawing.T, ordinate=Drawing.Y, width=1, spacing=1, offset=1, nb_points=500):
        super()
        self.output_shape = Embedding.MATRIX_SHAPE
        self.abscissa = abscissa
        self.ordinate = ordinate
        self.width = width
        self.spacing = spacing
        self.offset = offset
        self.nb_points = nb_points
    
    # TODO : Needs to sample only part of the points
    def embed(self, data):
        fen = 2 * self.width + 1 # Taille de la fenetre (nb colonnes dans la mat)

        stroke = data.interpolate(self.nb_points)
        if np.max(stroke[:,:2]) > 1 + TDA.INTERPOLATION_THRESHOLD or np.min(stroke[:,:2]) < 0 - TDA.INTERPOLATION_THRESHOLD:
            return None
        stroke = stroke[stroke[:,self.abscissa].argsort()] # Tri selon la variable abscisse
        shape = stroke.shape[0] # nb lignes
        mat = np.zeros((shape, fen)) # Initialisation de la matrice pour une stroke
        le = self.width * self.spacing 
        for i,depart in enumerate(range(le, shape - le, self.offset)):
            mat[i] = stroke[range(depart - le, depart + le + 1, self.spacing), self.ordinate]
        return mat[0:i+1]
    
    def get_data_shape(self):
        ws = self.width * self.spacing
        a = len(range(ws, self.nb_points - ws, self.offset))
        b = 2 * self.width + 1
        return (a, b, 1)

class Config:
    def __init__(self, embedding, evaluator, skip=False):
        # Data ==> Embedding ==> Reshape ==> Evaluate
        self.embedding = embedding
        self.evaluator = evaluator
        self.skip = skip

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
        self.results = []
    
    def init_data_storage(self):
        for path in self.datasets_path:
            if path not in self.stored_data:
                self.stored_data[path] = {"data":None}
            for line in self.configs:
                for config in line:
                    if config.embedding not in self.stored_data[path]:
                        self.stored_data[path][config.embedding.embedding_id] = None

    def run(self):
        self.results = []
        for line_index, line in enumerate(self.configs):
            self.results.append([])
            for config in line:
                if config.skip:
                    self.results[line_index].append(-1)
                    continue
                train_data = []
                train_labels = []
                for path in self.datasets_path:
                    # Extract data
                    if self.stored_data[path][config.embedding.embedding_id] is None:    
                        if self.stored_data[path]["data"] is None:
                            data = self.read_data(path)
                            if self.store_data:
                                self.stored_data[path]["data"] = data
                        else:
                            data = self.stored_data[path]["data"]
                        embedding = [config.embedding.embed(draw) for draw in data]
                        """
                        embedding = []
                        for draw in data:
                            try:
                                embedding.append(config.embedding.embed(draw))
                            except:
                                pass
                        """
                        if self.store_data:
                            self.stored_data[path][config.embedding.embedding_id] = embedding
                    else:
                        data = self.stored_data[path]["data"]
                        embedding = self.stored_data[path][config.embedding.embedding_id]
                    embedding = config.embedding.post_embedding(embedding)
                    train_labels = train_labels + [d.label for d in data]
                    train_data = train_data + embedding
                
                train_labels = [train_labels[i] for i,e in enumerate(train_data) if e is not None]
                train_data = [e for i,e in enumerate(train_data) if e is not None]
                
                X = config.evaluator.reshape(train_data, config.embedding.output_shape)
                y = np.array(train_labels)

                if config.evaluator.task == Evaluator.CLASSIFICATION:
                    n = X.shape[0]
                    lim = int(self.train_ratio * n)
                    shuffle = np.arange(0, n)
                    np.random.shuffle(shuffle)

                    X_train = X[shuffle][:lim]
                    y_train = y[shuffle][:lim]
                    X_test = X[shuffle][lim:]
                    y_test = y[shuffle][lim:]

                    config.evaluator.fit(X_train, y=y_train, X_val=X_test, y_val=y_test)
                    score = config.evaluator.accuracy(X_test, y_test)
                
                elif config.evaluator.task == Evaluator.CLUSTERING:
                    
                    config.evaluator.fit(X)
                    """
                    import sklearn.decomposition
                    pca = sklearn.decomposition.PCA()
                    pca.fit(X)
                    #from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
                    #lda = LinearDiscriminantAnalysis(n_components=2)
                    #lda.fit(X,y)
                    
                    trans = pca.transform(X)
                    import matplotlib.pyplot as plt

                    true_colors = np.copy(y)
                    colors = config.evaluator.predict(X)
                    for i,lab in enumerate(np.unique(true_colors)):
                        true_colors[true_colors==lab] = i
                    
                    for i,lab in enumerate(np.unique(colors)):
                        colors[colors==lab] = i
                    
                    colors = colors.astype(float)
                    true_colors = true_colors.astype(float)
                    
                    plt.figure(1)
                    plt.subplot(1, 2, 2)
                    plt.scatter(trans[:,0], trans[:,1], c=colors, s=2)
                    plt.subplot(1, 2, 1)
                    plt.scatter(trans[:,0], trans[:,1], c=true_colors, s=2)
                    plt.show()
                    """
                    score = config.evaluator.accuracy(X, y)
                
                print(type(config.embedding).__name__ + " - " + type(config.evaluator).__name__)
                print(score)
                self.results[line_index].append(score)
        
    
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
        
        res = [Drawing(draw,
                        do_link_strokes=self.do_link_strokes,
                        do_rescale=self.do_rescale,
                        link_steps=self.link_steps,
                        link_timestep=self.link_timestep) for draw in data]
        return [drawing for drawing in res if drawing.recognized]
    
    def latex_results(self, decimals=2):
        w = len(self.results[0])
        #h = len(self.results)
        results = np.around(np.array(self.results), decimals).astype(np.str)
        data = " \\\\\n".join(["EMBED & " + " & ".join(line) for line in results]) + " \\\\\n"
        final = "\\begin{table}[h]\n"
        final += "\\begin{tabular}{c|" + (w * "c") + "}\n"
        final += "\\hline\n"
        final += " & ".join([' '] + (['EVAL'] * w)) + " \\\\\n"
        final += "\\hline\n"
        final += data
        final += "\\hline\n"
        final += "\\end{tabular}\n"
        final += "\\end{table}"
        return final
    
    def show_curves(self, abscissa, evaluators):
        res = np.array(self.results)
        for i in range(res.shape[1]):
            plt.plot(abscissa, res[:,i], label=evaluators[i])
        plt.legend(loc="lower right")
        """
        plt.ylim(
            max(np.min(res) - 0.05, 0),
            min(np.max(res) + 0.05, 1)
        )
        """
        plt.ylim(0,1)
        plt.show()

if __name__ == '__main__':
    nb_classes = 2

    lr = LogisticRegressorEvaluator()
    svm = SVMEvaluator()
    signature = Signature(4, log=False)
    tda = TDA(abscissa=Drawing.T, ordinate=Drawing.X, width=2, spacing=1, offset=1, nb_points=500)
    sc = SpectralClusteringEvaluator(nb_classes)
    em = EMClusteringEvaluator(nb_classes)
    cnn = CNNEvaluator(tda.get_data_shape(), nb_classes, dropout=0.4)

    empty = Config(None, None, skip=True)

    config = Config(signature, lr)
    config3 = Config(signature, svm)
    config4 = Config(signature, em)
    config5 = Config(signature, sc, skip=False)

    spline = Spline(abscissa=Drawing.X, ordinate=Drawing.Y, degree=1, nb_knots=10)
    
    config6 = Config(spline, lr)
    config7 = Config(spline, svm)
    config8 = Config(spline, em)
    config9 = Config(spline, sc, skip=False)

    config10 = Config(tda, lr)
    config11 = Config(tda, svm)
    config12 = Config(tda, cnn)
    config13 = Config(tda, em)
    config14 = Config(tda, sc, skip=False)

    tester = Tester(
        ["../data/full_raw_axe.ndjson", "../data/full_raw_squirrel.ndjson", "../data/full_raw_sword.ndjson",
        "../data/full_raw_The Eiffel Tower.ndjson", "../data/full_raw_basketball.ndjson"][:nb_classes],
        [
            [config]
        ],
        store_data=True,
        nb_lines=2000,
        do_link_strokes=True,
        do_rescale=True
    )
    #cProfile.run('tester.run()', sort="cumtime")
    start = timer()
    tester.run()
    print(timer() - start)
    print(tester.results)
    
    exit(0)

# TODO :
# PCA for clustering ? Non pas la priorité
# Visualization for clusters ? mouais
# Storing results ! oui
# Add parameters to evaluators
# check if recognized in list comprehension of embedding