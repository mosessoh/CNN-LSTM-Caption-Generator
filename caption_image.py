import sys, getopt
from image_feature_cnn import forward_cnn
import tensorflow as tf
from model import Model

best_model_dir = 'best_model'

class Config(object):
    img_dim = 1024
    hidden_dim = embed_dim = 512
    max_epochs = 50
    batch_size = 256
    keep_prob = 0.75
    layers = 2
    model_name = 'model_keep=%.2f_batch=%d_hidden_dim=%d_embed_dim=%d_layers=%d' % (keep_prob, batch_size, hidden_dim, embed_dim, layers)

def main(argv):
    opts, args = getopt.getopt(argv, 'i:')
    for opt, arg in opts:
        if opt == '-i':
            img_path = arg

    config = Config()
    with tf.variable_scope('CNNLSTM') as scope:
        print '-'*20
        print 'Model info'
        print '-'*20
        model = Model(config)
        print '-'*20
    saver = tf.train.Saver()
    
    img_vector = forward_cnn(img_path)

    with tf.Session() as session:
        save_path = best_model_dir + '/model-37'
        saver.restore(session, save_path)
        print '2 Layer LSTM loaded'
        print 'Generating caption...'
        caption = model.generate_caption(session, img_vector)
        print 'Output:', caption

if __name__ == '__main__':
    main(sys.argv[1:])
