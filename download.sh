#! /bin/bash
cd data_files
if [ ! -f index2token.pkl ]; then
    wget https://googledrive.com/host/0B8-CPllE3RJqaHlRTWxxSTI3cW8/index2token.pkl
fi
if [ ! -f preprocessed_train_captions.pkl ]; then
    wget https://googledrive.com/host/0B8-CPllE3RJqaHlRTWxxSTI3cW8/preprocessed_train_captions.pkl
fi
if [ ! -f train_image_id2feature.pkl ]; then
    wget https://googledrive.com/host/0B8-CPllE3RJqaHlRTWxxSTI3cW8/train_image_id2feature.pkl
fi
if [ ! -f val_image_id2feature.pkl ]; then
    wget https://googledrive.com/host/0B8-CPllE3RJqaHlRTWxxSTI3cW8/val_image_id2feature.pkl
fi
cd ..
cd best_model
if [ ! -f model-37 ]; then
    wget https://googledrive.com/host/0B8-CPllE3RJqaHlRTWxxSTI3cW8/checkpoint
    wget https://googledrive.com/host/0B8-CPllE3RJqaHlRTWxxSTI3cW8/model-37
    wget https://googledrive.com/host/0B8-CPllE3RJqaHlRTWxxSTI3cW8/model-37.meta
fi