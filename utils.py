from easydict import EasyDict as edict
from pprint import pprint
import json
import argparse
import os
import sys

def parse_args():

    # Create a parser
    parser = argparse.ArgumentParser(description="Visual Speech Recognition")
    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')
    parser.add_argument('--config', default=None, type=str, help='Configuration file')
    parser.add_argument('--checkpoint', default=None, type=str, help='Model checkpoint file')
    parser.add_argument('--save_path', default="/mnt/disks/data/runs/", type=str, help='Top level directory to store logs and checkpoints for all runs')
    parser.add_argument('--train', default=False, type=bool, help='True if training')

    # Parse the arguments
    args = parser.parse_args()

    # Parse the configurations from the config json file provided
    try:
        if args.config is not None:
            with open(args.config, 'r') as config_file:
                config_args_dict = json.load(config_file)
        else:
            print("Add a config file using \'--config file_name.json\'", file=sys.stderr)
            exit(1)

    except FileNotFoundError:
        print("ERROR: Config file not found: {}".format(args.config), file=sys.stderr)
        exit(1)
    except json.decoder.JSONDecodeError:
        print("ERROR: Config file is not a proper JSON file!", file=sys.stderr)
        exit(1)

    config_args = edict(config_args_dict)
    config_args.test = True if config_args.test != 0 else False
    config_args.train = args.train
    config_args.save_path = args.save_path
    config_args.checkpoint = args.checkpoint
    

    pprint(config_args)
    print("\n")

    return config_args
