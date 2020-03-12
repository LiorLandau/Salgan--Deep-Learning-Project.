from tqdm import tqdm
import cv2
import glob
import os
import numpy as np
import theano
import lasagne
from Paths import HOME_DIR


def chunks(l, n):
    for i in xrange(0, len(l), n):
        yield l[i:i + n]


def load_weights(net, path, epochtoload):

    with np.load(HOME_DIR + path + "weights.npz".format(epochtoload)) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(net, param_values)


def predict(model, image_stimuli, num_epoch=None, name=None, path_output_maps=None):

    size = (image_stimuli.shape[1], image_stimuli.shape[0])
    blur_size = 5

    if image_stimuli.shape[:2] != (192, 256):
        image_stimuli = cv2.resize(image_stimuli, (256, 192), interpolation=cv2.INTER_AREA)

    blob = np.zeros((1, 3, 192, 256), theano.config.floatX)

    blob[0, ...] = (image_stimuli.astype(theano.config.floatX).transpose(2, 0, 1))

    result = np.squeeze(model.predict_use_theano(blob))
    saliency_map = (result * 255).astype(np.uint8)

    # resize back to original size
    saliency_map = cv2.resize(saliency_map, size, interpolation=cv2.INTER_CUBIC)
    # blur
    saliency_map = cv2.GaussianBlur(saliency_map, (blur_size, blur_size), 0)
    # clip again
    saliency_map = np.clip(saliency_map, 0, 255)
    if name is None:
        # When we use for testing, there is no file name provided.
        cv2.imwrite('./' + path_output_maps + '/validationRandomSaliencyPred_{:04d}.png'.format(num_epoch), saliency_map)
    else:
        cv2.imwrite(os.path.join(path_output_maps, name + '.jpg'), saliency_map)

def make_test(path_to_images, path_output_maps, model_to_test=None):
    list_img_files = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(path_to_images, '*'))]
    # Load Data
    list_img_files.sort()
    for image in tqdm(list_img_files, ncols=20):
        print os.path.join(path_to_images, image + '.jpg')
        img = cv2.cvtColor(cv2.imread(os.path.join(path_to_images, image + '.jpg'), cv2.IMREAD_COLOR),
                           cv2.COLOR_BGR2RGB)
        predict(model=model_to_test, image_stimuli=img, name=image, path_output_maps=path_output_maps)
