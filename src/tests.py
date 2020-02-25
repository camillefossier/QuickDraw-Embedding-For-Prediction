from config import *
import os

def generate_configs(embeddings, evaluators):
    return [[Config(em, ev) for ev in evaluators] for em in embeddings]

def spline_array(datasets):
    K = len(datasets)
    nb_knots = 15
    degree = 3
    s1 = Spline(abscissa=Drawing.T, ordinate=Drawing.X, nb_knots=nb_knots, degree=degree)
    s2 = Spline(abscissa=Drawing.T, ordinate=Drawing.Y, nb_knots=nb_knots, degree=degree)
    s3 = Spline(abscissa=Drawing.T, ordinate=Drawing.Z, nb_knots=nb_knots, degree=degree)
    s4 = Spline(abscissa=Drawing.T, ordinate=[Drawing.X, Drawing.Y], nb_knots=nb_knots, degree=degree)
    s5 = Spline(abscissa=Drawing.T, ordinate=[Drawing.X, Drawing.Y, Drawing.Z], nb_knots=nb_knots, degree=degree)
    s6 = Spline(abscissa=Drawing.X, ordinate=Drawing.Y, nb_knots=nb_knots, degree=degree)
    s7 = Spline(abscissa=Drawing.Y, ordinate=Drawing.X, nb_knots=nb_knots, degree=degree)

    e1 = LogisticRegressorEvaluator()
    e2 = SVMEvaluator()
    e31 = CNNEvaluator(input_shape=s1.get_data_shape(), num_classes=K, filter_size=(2,2))
    e32 = CNNEvaluator(input_shape=s4.get_data_shape(), num_classes=K, filter_size=(2,2))
    e33 = CNNEvaluator(input_shape=s5.get_data_shape(), num_classes=K, filter_size=(2,2))
    e4 = SpectralClusteringEvaluator(K)
    e5 = EMClusteringEvaluator(nb_clusters=K)

    configs = generate_configs(
        [s1, s2, s3, s4, s5, s6, s7],
        [e1, e2, e31, e4, e5]
    )

    configs[3][2].evaluator = e32
    configs[4][2].evaluator = e33
    
    tester = Tester(datasets, configs, store_data=True, nb_lines=2000, do_link_strokes=True, do_rescale=True)
    tester.run()
    return tester

def spline_knots(datasets, knots):
    K = len(datasets)
    e1 = LogisticRegressorEvaluator()
    e2 = SVMEvaluator()
    #e3 = CNNEvaluator(input_shape=(10,10,1), num_classes=K)
    e4 = SpectralClusteringEvaluator(K)
    e5 = EMClusteringEvaluator(nb_clusters=K)
    
    splines = [Spline(abscissa=Drawing.X, ordinate=Drawing.Y, nb_knots=knot) for knot in knots] 
    configs = generate_configs(
        splines,
        [e1, e2, e4, e5]
    )
    tester = Tester(
        datasets,
        configs,
        store_data=True,
        nb_lines=1000,
        do_link_strokes=True,
        do_rescale=True
    )
    tester.run()
    return tester

def signature_degree(datasets, degrees):
    K = len(datasets)
    e1 = LogisticRegressorEvaluator()
    e2 = SVMEvaluator()
    #e3 = CNNEvaluator(input_shape=(10,10,1), num_classes=K)
    e4 = SpectralClusteringEvaluator(K)
    e5 = EMClusteringEvaluator(nb_clusters=K)
    
    signatures = [Signature(4, nb_to_keep=deg, embedding_id="signature") for deg in degrees] 
    configs = generate_configs(
        signatures,
        [e1, e2]
    )
    tester = Tester(
        datasets,
        configs,
        store_data=True,
        nb_lines=1000,
        do_link_strokes=True,
        do_rescale=True
    )
    tester.run()
    return tester

if __name__ == '__main__':
    datasets = [os.path.join('..\\data', filename) for filename in os.listdir('../data')]

    sample_files = np.random.choice(datasets, 5, replace=False).tolist()

    spline = spline_array(datasets[:2])
    #sig_graph = signature_degree(datasets, list(range(10,341,10)))
    #sig_graph.show_curves(list(range(10,341,10)), evaluators=("Regression Logistique", "SVM"))
    graph = spline_knots(datasets, list(range(2,102,5)))
    graph.show_curves(list(range(2,102,5)), evaluators=["LR", "SVM", "Spectral", "EM"])
    print(spline.latex_results(3))