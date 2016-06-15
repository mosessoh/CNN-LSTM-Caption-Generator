import numpy as np
import os, sys, getopt, time
import cPickle as pickle
 
# Main path to your caffe installation
caffe_root = '/home/mosessoh/caffe/'
# Model prototxt file
model_prototxt = caffe_root + 'models/bvlc_googlenet/deploy.prototxt'
# Model caffemodel file
model_trained = caffe_root + 'models/bvlc_googlenet/bvlc_googlenet.caffemodel'
# File containing the class labels
imagenet_labels = caffe_root + 'data/ilsvrc12/synset_words.txt'
# Path to the mean image (used for input processing)
mean_path = caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
# Name of the layer we want to extract
layer_name = 'pool5/7x7_s1'

os.environ['GLOG_minloglevel'] = '3'
 
sys.path.insert(0, caffe_root + 'python')
import caffe

# Setting this to CPU, but feel free to use GPU if you have CUDA installed
caffe.set_mode_cpu()
# Loading the Caffe model, setting preprocessing parameters
net = caffe.Classifier(model_prototxt, model_trained,
                       mean=np.load(mean_path).mean(1).mean(1),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))


def forward_cnn(image_path):
    image_path = image_path.strip()
    input_image = caffe.io.load_image(image_path)
    prediction = net.predict([input_image], oversample=False)
    image_vector = net.blobs[layer_name].data[0].reshape(1,-1)
    print 'CNN forward pass completed'
    return image_vector

 
def main(argv):
    inputfile = ''
    outputfile = ''
 
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print 'caffe_feature_extractor.py -i <inputfile> -o <outputfile>'
        sys.exit(2)
 
    for opt, arg in opts:
        if opt == '-h':
            print 'caffe_feature_extractor.py -i <inputfile> -o <outputfile>'
            sys.exit()
        elif opt in ("-i"):
            inputfile = arg
        elif opt in ("-o"):
            outputfile = arg
 
    print 'Reading images from "', inputfile
    print 'Writing vectors to "', outputfile
 
    # Setting this to CPU, but feel free to use GPU if you have CUDA installed
    caffe.set_mode_cpu()
    # Loading the Caffe model, setting preprocessing parameters
    net = caffe.Classifier(model_prototxt, model_trained,
                           mean=np.load(mean_path).mean(1).mean(1),
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(256, 256))
 
    # Loading class labels
    with open(imagenet_labels) as f:
        labels = f.readlines()
 
    # This prints information about the network layers (names and sizes)
    # You can uncomment this, to have a look inside the network and choose which layer to print
    #print [(k, v.data.shape) for k, v in net.blobs.items()]
    #exit()

    if inputfile == 'train_images.txt':
        num_imgs = 82783
    elif inputfile == 'val_images.txt':
        num_imgs = 40504
    elif inputfile == 'sample_images.txt':
        num_imgs = 141
 
    # Processing one image at a time, printing predictions and writing the vector to a file
    start = time.time()
    counter = 1
    with open(inputfile, 'r') as reader:
        for image_path in reader:
            print 'Processing %d of %d' % (counter, num_imgs)
            if counter % 10 == 0:
                print 'Time elapsed (min): %.1f' % (time.time() - start)

            image_path = image_path.strip()
            input_image = caffe.io.load_image(image_path)
            prediction = net.predict([input_image], oversample=False)
            image_vector = net.blobs[layer_name].data[0].reshape(1,-1)

            image_picklename = os.path.splitext(image_path)[0] + '.p'
            pickle.dump(image_vector, open(image_picklename,'w'))
            counter += 1

    print 'Time elapsed (s): %.4f' % (time.time() - start)
    print 'Avg Time per Image (s): %.4f' % ((time.time() - start)/num_imgs)

 
# if __name__ == "__main__":
#     main(sys.argv[1:])