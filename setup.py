"""The setup module."""

# -*- coding: utf-8 -*-

import glob
import os
import setuptools

if __name__ == '__main__':
    setuptools.setup()
    # Setup for Bluebert model
    for source_name in glob.glob("./**/convert_bert_original_tf_checkpoint_to_pytorch.py"):
        python_file_path, fullname = os.path.split(source_name)
    for source_name in glob.glob("./**/bluebert/**/bert_model.ckpt"):
        bluebert_repo_path, fullname = os.path.split(source_name)
    command = python_file_path+"/convert_bert_original_tf_checkpoint_to_pytorch.py "\
              "tf_checkpoint_path="+bluebert_repo_path+"/bert_model.ckpt "\
              "bert_config_file="+bluebert_repo_path+"/bert_config.json "\
              "pytorch_dump_path="+bluebert_repo_path+"/pytorch_model.bin"
    os.system(command)
    for source_name in glob.glob("./**/bluebert/**/bert-config.json"):
        path, fullname = os.path.split(source_name)
        target_name = os.path.join(path, 'config.json')
        os.rename(source_name, target_name)
