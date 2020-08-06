import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from cif3r.models.train_siamese import make_pairs, get_pairs, process_path
from cif3r.features.preprocessing import train_val_split
from cif3r.models.siamese_training import feature_xtract as feature_model
from cif3r.features.preprocessing import datagen
from sklearn.manifold import TSNE
import random
import numpy as np


clf = tf.keras.models.load_model(
    "/home/ubuntu/CIf3R/models/UTK_siamese.h5", custom_objects={"tf": tf}
)


def show_model_output():
    pairs, labels = make_pairs(minority_cls_count=2, total_pairs=1)
    samples = [process_path(imgs, label) for imgs, label in zip(pairs, labels)]
    print(len(samples[1]))
    samples = random.choices(samples, k=8)
    img1 = [sublst[0] for sublst in samples]
    img2 = [sublst[1] for sublst in samples]
    labels = [sublst[-1] for sublst in samples]
    print("Predicting pair classes:")
    pred_sim = clf.predict([img1, img2])
    fig, m_axs = plt.subplots(2, 8, figsize=(16, 8))
    for c_a, c_b, c_d, p_d, (ax1, ax2) in zip(img1, img2, labels, pred_sim, m_axs.T):
        ax1.imshow(c_a[:, :, 0])
        ax1.set_title("Image A\n Actual: %3.0f%%" % (100 * c_d))
        ax1.axis("off")
        ax2.imshow(c_b[:, :, 0])
        ax2.set_title("Image B\n Predicted: %3.0f%%" % (100 * p_d))
        ax2.axis("off")
    plt.savefig("/home/ubuntu/CIf3R/reports/figures/img-similarity-siam.png")
    return fig


def plot_tsne(university: str = "UTK"):
    print("Getting test data for TSNE")

    test_df = datagen(university, balance_method="undersampling")

    obj_categories = ["paper", "cans", "plastic", "trash"]
    colors = plt.cm.rainbow(np.linspace(0, 1, 10))
    imagegen = tf.keras.preprocessing.image.ImageDataGenerator()
    x_test = imagegen.flow_from_dataframe(
        test_df, target_size=(105, 105), batch_size=128
    )

    plt.figure(figsize=(10, 10))
    tsne_obj = TSNE(
        n_components=2,
        init="pca",
        random_state=101,
        method="barnes_hut",
        n_iter=500,
        verbose=2,
        n_jobs=-1,
    )
    x_test_features = feature_model.predict(x_test, verbose=True)
    y_test = test_df["class"].to_numpy
    tsne_features = tsne_obj.fit_transform(x_test_features)
    for c_group, (c_color, c_label) in enumerate(zip(colors, obj_categories)):
        plt.scatter(
            tsne_features[np.where(y_test == c_group), 0],
            tsne_features[np.where(y_test == c_group), 1],
            marker="o",
            color=c_color,
            linewidth="1",
            alpha=0.8,
            label=c_label,
        )
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title("t-SNE on Testing Samples")
    plt.legend(loc="best")
    plt.savefig(f"/home/ubuntu/CIf3R/reports/figures/{university}_tsne.png")


# show_model_output()
plot_tsne()
