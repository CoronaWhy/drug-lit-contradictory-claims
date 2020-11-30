#!/bin/bash
#
#SBATCH --job-name=train_contra_claims
#SBATCH --partition=rbaltman,owners
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=2-

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/share/software/user/open/cudnn/7.6.5/lib64"
#pwd
pip install -e .
python -m contradictory_claims --train --report --roberta --batch_size=2 --epochs=3 --learning_rate=0.000001
python -m contradictory_claims --train --report --roberta --batch_size=2 --epochs=3 --learning_rate=0.000001
python -m contradictory_claims --train --report --roberta --batch_size=2 --epochs=3 --learning_rate=0.000001
python -m contradictory_claims --train --report --roberta --batch_size=2 --epochs=3 --learning_rate=0.000001
python -m contradictory_claims --train --report --roberta --batch_size=2 --epochs=3 --learning_rate=0.000001
python -m contradictory_claims --train --report --roberta --batch_size=2 --epochs=3 --learning_rate=0.000001
python -m contradictory_claims --train --report --roberta --batch_size=2 --epochs=3 --learning_rate=0.000001
python -m contradictory_claims --train --report --roberta --batch_size=2 --epochs=3 --learning_rate=0.000001
python -m contradictory_claims --train --report --roberta --batch_size=2 --epochs=3 --learning_rate=0.000001
python -m contradictory_claims --train --report --roberta --batch_size=2 --epochs=3 --learning_rate=0.000001
python -m contradictory_claims --train --report --roberta --batch_size=2 --epochs=3 --learning_rate=0.000001
python -m contradictory_claims --train --report --roberta --batch_size=2 --epochs=3 --learning_rate=0.000001
python -m contradictory_claims --train --report --roberta --batch_size=2 --epochs=3 --learning_rate=0.000001
python -m contradictory_claims --train --report --roberta --batch_size=2 --epochs=3 --learning_rate=0.000001
python -m contradictory_claims --train --report --roberta --batch_size=2 --epochs=3 --learning_rate=0.000001
python -m contradictory_claims --train --report --roberta --batch_size=2 --epochs=3 --learning_rate=0.000001
