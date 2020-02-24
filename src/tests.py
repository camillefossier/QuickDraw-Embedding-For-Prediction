from config import *

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
    e3 = CNNEvaluator(input_shape=(10,10,1), num_classes=K)
    e4 = SpectralClusteringEvaluator(K)
    e5 = EMClusteringEvaluator(nb_clusters=K)

    configs = generate_configs(
        [s1, s2, s3, s4, s5, s6, s7],
        [e1, e2, e3, e4, e5]
    )

    for config_line in configs:
        config_line[2].skip = True
    
    tester = Tester(datasets, configs, store_data=True, nb_lines=1000, do_link_strokes=True, do_rescale=True)
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
    datasets = [
        "../data/full_raw_axe.ndjson",
        "../data/full_raw_squirrel.ndjson",
        "../data/full_raw_sword.ndjson",
        "../data/full_raw_The Eiffel Tower.ndjson",
        "../data/full_raw_basketball.ndjson"
    ]

    #sig_graph = signature_degree(datasets, list(range(10,341,10)))
    #sig_graph.show_curves(list(range(10,341,10)), evaluators=("Regression Logistique", "SVM"))
    graph = spline_knots(datasets, list(range(2,102,5)))
    graph.show_curves(list(range(2,102,5)), evaluators=["LR", "SVM", "Spectral", "EM"])
    spline = spline_array(datasets)
    print(spline.latex_results(3))