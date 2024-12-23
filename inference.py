import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="0")
    parser.add_argument('--model_path', type=str, default="0015")
    opt = parser.parse_args()

    return opt
args = parse_args()


if __name__ == '__main__':
    