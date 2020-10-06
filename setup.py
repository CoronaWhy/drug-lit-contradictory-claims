"""The setup module."""

# -*- coding: utf-8 -*-

import os
import setuptools

if __name__ == '__main__':
    setuptools.setup()

	#Setup for Bluebert model
	os.system("convert_bert_original_tf_checkpoint_to_pytorch.py"\
				"--tf_checkpoint_path=bluebert/bert_model.ckpt"\
				"--bert_config_file=bluebert/bert_config.json"\
				"--pytorch_dump_path=bluebert/pytorch_model.bin")
