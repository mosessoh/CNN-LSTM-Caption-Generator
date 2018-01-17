#! /bin/bash
cd data_files
if [ ! -f index2token.pkl ]; then
    wget https://googledrive.com/host/0B8-CPllE3RJqaHlRTWxxSTI3cW8/index2token.pkl
fi
if [ ! -f preprocessed_train_captions.pkl ]; then
    wget https://s3.amazonaws.com/cnn-lstm-caption-generator/preprocessed_train_captions.pkl
fi
if [ ! -f train_image_id2feature.pkl ]; then
    wget https://s3.amazonaws.com/cnn-lstm-caption-generator/train_image_id2feature.pkl
fi
if [ ! -f val_image_id2feature.pkl ]; then
    wget https://s3.amazonaws.com/cnn-lstm-caption-generator/val_image_id2feature.pkl
fi
cd ..
cd best_model
if [ ! -f model-37 ]; then
    wget https://s3.amazonaws.com/cnn-lstm-caption-generator/checkpoint
    wget https://s3.amazonaws.com/cnn-lstm-caption-generator/model-37
    wget https://s3.amazonaws.com/cnn-lstm-caption-generator/model-37.meta
fi