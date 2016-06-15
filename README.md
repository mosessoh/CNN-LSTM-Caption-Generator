## Learning CNN-LSTM Architectures for Image Caption Generation

This code contains a Tensorflow implementation of the CNN-LSTM architecture used to attain state-of-the-art performance on the MSCOCO dataset. We achieve a BLEU-4 score of 24.4 and CIDEr score of 81.7 compared to 27.7 and 85.5 by Google's implementation. Qualitative analysis of the generated captions indicate that the model is able to sensibly caption a wide variety of images from the MSCOCO dataset.

### Demo instructions

To try a demo of our best trained model, first ensure that Caffe is installed on your computer and that you have downloaded the GoogleNet model using these [instructions](http://www.marekrei.com/blog/transforming-images-to-feature-vectors/). You'll also need Tensorflow 0.8 installed. Then, run:

    ./download.sh

which will retrive all pickled data files (graciously shared by [Satoshi](http://t-satoshi.blogspot.com/2015/12/image-caption-generation-by-cnn-and-lstm.html) in his chainer implementation.) and the Tensorflow saved model created in this project needed to run the demo. This requires around 180MB of disk space. The 'caption_image.py' file contains all the code needed to load and use the saved model. To run the demo, do:

    python caption_image.py -i <path_to_image>

We have included a demo pizza image at images/pizza.jpg to sanity check your installation. Running `python caption_image.py -i images/pizza.jpg` produces the caption "a pizza with cheese and cheese on a table". It's not perfect, but still pretty cool!

### Other files
`model.py` contains the `Model` class that contains the CNN-LSTM architecture (using Tensorflow's dynamic_rnn API) and various helper functions for generating captions. `evaluate_captions.py` is a helper script to generate aggregated JSON files that can then be used for hyperparameter tuning. `image_feature_cnn.py` contains the helper functions we use to load up the GoogleNet batch normalization CNN model and turn images into 1024 x 1 vectors.