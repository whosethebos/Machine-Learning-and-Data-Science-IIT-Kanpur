import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import os.path
from sklearn.metrics import f1_score, accuracy_score
from model import create_model
from sklearn.manifold import TSNE
from keras.models import Model
from keras.layers import Input, Layer
from data import triplet_generator
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
import matplotlib.patches as patches
from align import AlignDlib
import TripletLossLayer as TripletLossLayer
import IdentityMetadata as IdentityMetadata


def load_metadata(path):
    metadata = []
    for i in sorted(os.listdir(path)):
        for f in sorted(os.listdir(os.path.join(path, i))):
            # Check file extension. Allow only jpg/jpeg' files.
            ext = os.path.splitext(f)[1]
            if ext == '.jpg' or ext == '.jpeg':
                metadata.append(IdentityMetadata.IdentityMetadata(path, i, f))
    return np.array(metadata)

def load_image(path):
    img = cv2.imread(path, 1)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    return img[...,::-1]


def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))


def main(predict_eg):
    nn4_small2 = create_model()
    # Input for anchor, positive and negative images
    input_a = Input(shape=(96, 96, 3))
    input_p = Input(shape=(96, 96, 3))
    input_n = Input(shape=(96, 96, 3))
    # Output for anchor, positive and negative embedding vectors
    emb_a = nn4_small2(input_a)
    emb_p = nn4_small2(input_p)
    emb_n = nn4_small2(input_n)
    # Layer that computes the triplet loss from anchor, positive and negative embedding vectors
    triplet_loss_layer = TripletLossLayer.TripletLossLayer(alpha=0.2, name='triplet_loss_layer')([emb_a, emb_p, emb_n])
    # Model that can be trained with anchor, positive negative images
    nn4_small2_train = Model([input_a, input_p, input_n], triplet_loss_layer)
    # triplet_generator() creates a generator that continuously returns
    # ([a_batch, p_batch, n_batch], None) tuples where a_batch, p_batch
    # and n_batch are batches of anchor, positive and negative RGB images
    # each having a shape of (batch_size, 96, 96, 3).
    generatorImage = triplet_generator()
    nn4_small2_train.compile(loss=None, optimizer='adam')
    nn4_small2_train.fit_generator(generatorImage, epochs=10, steps_per_epoch=100)
    # nn4_small2_train.fit_generator(generatorImage, epochs=2, steps_per_epoch=100)
    # Please note that the current implementation of the generator only generates
    # random image data. The main goal of this code snippet is to demonstrate
    # the general setup for model training. In the following, we will anyway
    # use a pre-trained model so we don't need a generator here that operates
    # on real training data. I'll maybe provide a fully functional generator
    # later.
    nn4_small2_pretrained = create_model()
    nn4_small2_pretrained.load_weights('weights/nn4.small2.v1.h5')
    metadata = load_metadata('images')
    # Initialize the OpenFace face alignment utility
    alignment = AlignDlib('models/landmarks.dat')
    # Load any random image
    jc_orig = load_image(metadata[77].image_path())
    # Detect face and return bounding box
    bb = alignment.getLargestFaceBoundingBox(jc_orig)
    # Transform image using specified face landmark indices and crop image to 96x96
    jc_aligned = alignment.align(96, jc_orig, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
    # Show original image
    plt.subplot(131)
    plt.imshow(jc_orig)
    plt.title("Orig Image")
    # plt.show()
    # Show original image with bounding box
    plt.subplot(132)
    plt.imshow(jc_orig)
    plt.title("with BB")
    # plt.show()
    plt.gca().add_patch(patches.Rectangle((bb.left(), bb.top()), bb.width(), bb.height(), fill=False, color='red'))
    # Show aligned image
    plt.subplot(133)
    plt.imshow(jc_aligned);
    plt.title("Aligned Image")
    plt.show()

    def align_image(img):
        return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img),
                               landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)


    embedded = np.zeros((metadata.shape[0], 128))
    for i, m in enumerate(metadata):
        img = load_image(m.image_path())
        img = align_image(img)
        # scale RGB values to interval [0,1]
        img = (img / 255.).astype(np.float32)
        # obtain embedding vector for image
        embedded[i] = nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]


    def show_pair(idx1, idx2):
        plt.figure(figsize=(8,3))
        plt.suptitle(f'Distance = {distance(embedded[idx1], embedded[idx2]):.2f}')
        plt.subplot(121)
        plt.imshow(load_image(metadata[idx1].image_path()))
        # plt.show()
        plt.subplot(122)
        plt.imshow(load_image(metadata[idx2].image_path()));
        plt.show()

    show_pair(77, 78)
    show_pair(77, 50)
    distances = []  # squared L2 distance between pairs
    identical = []  # 1 if same identity, 0 otherwise
    num = len(metadata)

    for i in range(num - 1):
        for j in range(i + 1, num):
            distances.append(distance(embedded[i], embedded[j]))
            identical.append(1 if metadata[i].name == metadata[j].name else 0)

    distances = np.array(distances)
    identical = np.array(identical)
    thresholds = np.arange(0.3, 1.0, 0.01)
    #F1 Score
    f1_scores = [f1_score(identical, distances < t) for t in thresholds]
    acc_scores = [accuracy_score(identical, distances < t) for t in thresholds]
    opt_idx = np.argmax(f1_scores)
    # Threshold at maximal F1 score
    opt_tau = thresholds[opt_idx]
    # Accuracy at maximal F1 score
    opt_acc = accuracy_score(identical, distances < opt_tau)
    # Plot F1 score and accuracy as function of distance threshold
    plt.plot(thresholds, f1_scores, label='F1 score');
    plt.plot(thresholds, acc_scores, label='Accuracy');
    plt.axvline(x=opt_tau, linestyle='--', lw=1, c='lightgrey', label='Threshold')
    plt.title(f'Accuracy at threshold {opt_tau:.2f} = {opt_acc:.3f}');
    plt.xlabel('Distance threshold')
    plt.legend();
    plt.show()
    dist_pos = distances[identical == 1]
    dist_neg = distances[identical == 0]
    plt.figure(figsize=(12,4))
    plt.subplot(121)
    plt.hist(dist_pos)
    plt.axvline(x=opt_tau, linestyle='--', lw=1, c='lightgrey', label='Threshold')
    plt.title('Distances (+ve pairs)')
    plt.legend();
    plt.subplot(122)
    plt.hist(dist_neg)
    plt.axvline(x=opt_tau, linestyle='--', lw=1, c='lightgrey', label='Threshold')
    plt.title('Distances (-ve pairs)')
    plt.legend();
    plt.show()
    targets = np.array([m.name for m in metadata])
    encoder = LabelEncoder()
    encoder.fit(targets)
    # Numerical encoding of identities
    y = encoder.transform(targets)
    train_idx = np.arange(metadata.shape[0]) % 2 != 0
    test_idx = np.arange(metadata.shape[0]) % 2 == 0
    # 50 train examples of 10 identities (5 examples each)
    X_train = embedded[train_idx]
    # 50 test examples of 10 identities (5 examples each)
    X_test = embedded[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    svc = LinearSVC()
    knn.fit(X_train, y_train)
    svc.fit(X_train, y_train)
    acc_knn = accuracy_score(y_test, knn.predict(X_test))
    acc_svc = accuracy_score(y_test, svc.predict(X_test))
    print(f'KNN accuracy = {acc_knn}, SVM accuracy = {acc_svc}')
    warnings.filterwarnings('ignore')
    example_idx = predict_eg
    print("example_idx",example_idx)
    example_image = load_image(metadata[test_idx][example_idx].image_path())
    example_prediction = svc.predict([embedded[test_idx][example_idx]])
    example_identity = encoder.inverse_transform(example_prediction)[0]
    plt.imshow(example_image)
    plt.title(f'Recognized as {example_identity}');
    plt.show()
    X_embedded = TSNE(n_components=2).fit_transform(embedded)

    for i, t in enumerate(set(targets)):
        idx = targets == t
        plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=t)

    plt.legend(bbox_to_anchor=(1, 1));
    plt.show()



