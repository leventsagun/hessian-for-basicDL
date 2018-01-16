from sklearn import datasets
from sklearn import preprocessing


def generate_data(args):
    if args.data_type == 'blob':
        inputs, targets = generate_blobs(2 * args.num_samples, args.input_dim, args.cov_factor, args.num_classes, args.data_seed)
    elif args.data_type == 'circle':
        inputs, targets = generate_circles(2 * args.num_samples, args.data_seed)
    elif args.data_type == 'moon':
        inputs, targets = generate_moons(2 * args.num_samples, args.data_seed)
    else:
        raise Exception('Unknown --data-type: {}'.format(args.data_type))

    encoder = preprocessing.OneHotEncoder()
    targets_nx1 = targets.reshape(-1, 1)
    targets_onehot = encoder.fit_transform(targets_nx1).toarray()
    inputs_scaled = preprocessing.scale(inputs)

    inputs_train, inputs_test = inputs_scaled[:args.num_samples], inputs_scaled[args.num_samples:]
    targets_train, targets_test = targets_onehot[:args.num_samples], targets_onehot[args.num_samples:]
    return inputs_train, targets_train, inputs_test, targets_test


def generate_blobs(num_samples, input_dim, cov_factor, num_classes, seed=None):
    """N dimensional features in two classes (Gaussian blobs)."""
    inputs, targets = datasets.make_blobs(n_samples=num_samples, n_features=input_dim, cluster_std=cov_factor, random_state=seed,
                                          centers=num_classes)
    return inputs, targets


def generate_circles(num_samples, seed=None, noise=0.1, factor =0.7):
    """Generate 2D features in two classes (concentric circles).
    
    http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html
    
    :param noise: Standard deviation of Gaussian noise added to the data 
    :param factor: Scale factor between inner and outer circle
    """
    inputs, targets = datasets.make_circles(n_samples=num_samples, shuffle=True, noise=noise, random_state=seed, factor=factor)
    return inputs, targets


def generate_moons(num_samples, seed=None, noise=0.1):
    """Generate 2D features in two classes (up and down crescent moons).
    
    http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html
     
    :param noise: Standard deviation of Gaussian noise added to the data
    """
    inputs, targets = datasets.make_moons(n_samples=num_samples, shuffle=True, noise=noise, random_state=seed)
    return inputs, targets
