#!/usr/bin/env python3
# # this is a runner for myAutoGrad examples
# usage: python3 run.py [download|train|test] [mnist|...]
# download: downloads datasets from internet
# train: compile run the corresponding training program
import os
import sys
def system(cmd):
    print(f"running: {cmd}")
    os.system(cmd)
download_url = {
    'mnist': 'https://github.com/wehrley/Kaggle-Digit-Recognizer/archive/refs/heads/master.zip',
}

train_cmd = {
    'mnist': './build/train_mnist load {parameter_file} train',
}

validate_cmd = {
    'mnist': './build/train_mnist load {parameter_file} validate',
}
parameter_file = {
    'mnist': 'out/mnist_model_params_interrupt.txt',
}
if len(sys.argv) < 3:
    print("usage: python3 run.py [download|compile|train|validate] [mnist|...]")
    sys.exit(1)
action = sys.argv[1]
example = sys.argv[2]

if action == 'download':
    if example not in download_url:
        print(f"no download url for {example}")
        sys.exit(1)
    url = download_url[example]
    system(f"wget {url} -O testcases/{example}.zip")
    system(f"unzip testcases/{example}.zip -d testcases/{example}")
    system(f"rm testcases/{example}.zip")
    print(f"downloaded and extracted {example} dataset")
elif action == 'compile':
    train_prog = f'./test/{example}.cpp'
    if not os.path.exists(train_prog):
        print(f"no training program for {example}")
        sys.exit(1)
    system(f"g++ -O3 {train_prog} -o build/train_{example}")
elif action == 'train':
    train_exec = f'./build/train_{example}'
    if not os.path.exists(train_exec):
        print(f"no compiled training executable for {example}, please run 'compile' first")
        sys.exit(1)
    system(train_cmd[example].format(parameter_file=parameter_file[example]))
elif action == 'validate':
    validate_exec = f'./build/train_{example}'
    if not os.path.exists(validate_exec):
        print(f"no compiled validation executable for {example}, please run 'compile' first")
        sys.exit(1)
    system(validate_cmd[example].format(parameter_file=parameter_file[example]))