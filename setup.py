"""The setup module."""

# -*- coding: utf-8 -*-

import os

import setuptools

if __name__ == '__main__':
    setuptools.setup()
    # Setup for Bluebert model
    # TODO: Searching "C:/" works on the local system but fails in tox. How to make this universal?
    for root, _dirs, files in os.walk("C:/"):
        for name in files:
            if name == 'convert_bert_original_tf_checkpoint_to_pytorch.py':
                python_file_path = os.path.abspath(root)
                break
    for root, dirs, _files in os.walk("C:/"):
        for name in dirs:
            if name == 'bluebert':
                bluebert_repo_path = os.path.abspath(os.path.join(root, name))
                break
    command = "python " + python_file_path + "/convert_bert_original_tf_checkpoint_to_pytorch.py "\
              "--tf_checkpoint_path=" + bluebert_repo_path + "/bert_model.ckpt "\
              "--bert_config_file=" + bluebert_repo_path + "/bert_config.json "\
              "--pytorch_dump_path=" + bluebert_repo_path + "/pytorch_model.bin"
    #TODO: Is there a way around this?
    os.system(command)  #noqa: S605
    source_file = os.path.join(bluebert_repo_path, "bert_config.json")
    target_file = os.path.join(bluebert_repo_path, "config.json")
    shutil.copy(source_file, target_file)
