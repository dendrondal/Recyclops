import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from cif3r.models.train_siamese import make_pairs, get_pairs, process_path
from cif3r.features.preprocessing import train_val_split
from sklearn.manifold import TSNE


clf = tf.keras.models.load_model('/home/ubuntu/CIf3R/models/UTK_siamese.h5', custom_objects={'tf': tf})

def show_model_output():
    pairs, labels = make_pairs(minority_cls_count=2, total_pairs=8)
    img1, img2, labels = map(process_path, [pairs, labels])
    pred_sim = clf.predict([img1, img2])
    fig, m_axs = plt.subplots(2, img1.shape[0], figsize = (12, 8))
    for c_a, c_b, c_d, p_d, (ax1, ax2) in zip(img1, img2, labels, pred_sim, m_axs.T):
        ax1.imshow(c_a[:,:,0])
        ax1.set_title('Image A\n Actual: %3.0f%%' % (100*c_d))
        ax1.axis('off')
        ax2.imshow(c_b[:,:,0])
        ax2.set_title('Image B\n Predicted: %3.0f%%' % (100*p_d))
        ax2.axis('off')
    plt.savefig('/home/ubuntu/CIf3R/reports/figures/img-similarity-siam.png')
    return fig


def X_test(university:str='UTK')
    _, X_test, _, y_test = train_val_split('UTK', test_size=0.3)
    obj_categories = ['paper', 'cans', 'plastic', 'trash']
    colors = plt.cm.rainbow(np.linspace(0, 1, 10))
    plt.figure(figsize=(10, 10))

    for c_group, (c_color, c_label) in enumerate(zip(colors, obj_categories)):
        plt.scatter(tsne_features[np.where(y_test == c_group), 0],
                    tsne_features[np.where(y_test == c_group), 1],
                    marker='o',
                    color=c_color,
                    linewidth='1',
                    alpha=0.8,
                    label=c_label)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('t-SNE on Testing Samples')
    plt.legend(loc='best')
    plt.savefig(f'/home/ubuntu/CIf3R/reports/figures/{university}_tsne.png')

