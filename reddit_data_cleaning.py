'''Minimal data cleaning, and splitting into train/test/dev'''

import random
import json
import re
from os.path import join
from os import makedirs, getcwd
import argparse
from utils import delete_dir
def remove_tags(sentence):
    return re.sub('\[[^\]]*\]','',sentence).strip()

def write(path,data):
    with open(path, 'w') as f:
        for d in data:
            f.write('<|endoftext|>')
            f.write(d)
        f.write('<|endoftext|>')

def create_dataset(data_source,
                output_dir,
                required_in_title = None,
                upvoted = False,
                skip_first = 0,
                bad_words = None,
                ):
    with open(data_source) as f:
        data = json.load(f)
    delete_dir(output_dir)
    makedirs(output_dir)
    data = [x for x in data if ("selftext" not in x) or (not x["selftext"])]
    if required_in_title is not None:
        for required in required_in_title:
            data  = [x for x in data if required in x["title"].lower()]
    if upvoted:
        data = [x for x in data if x["score"] > 1]
    data = [x["title"] for x in data]
    data = data[skip_first:] # Removing the first 1000 older examples
    # Trying to remove most prompts that directly ask to write or describe something
    if bad_words is not None:
        for word in bad_words:
            data = [x for x in data if (word not in x.lower())]
    data = [remove_tags(x) for x in data]
    data = list(set(data))  # removing reposts
    split_1 = int(len(data)*.9)
    split_2 = int(len(data)*.95)
    random.shuffle(data)

    data_train = data[:split_1]
    print("Train dataset of length",len(data_train))
    data_test = data[split_1:split_2]
    print("Test dataset of length",len(data_test))
    data_val = data[split_2:]
    print("Dev dataset of length",len(data_val))

    write(join(output_dir,"train.txt"), data_train)
    write(join(output_dir,"test.txt"), data_test)
    write(join(output_dir,"dev.txt"), data_val)
    print("data saved in",join(getcwd(),output_dir))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_source', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--required_in_title', nargs = '+',type=str,default=None)
    parser.add_argument('--bad_words', nargs = '+',type=str,default=None)
    parser.add_argument('--upvoted', default=False, action='store_true')
    parser.add_argument('--skip_first', type=int)

    args = parser.parse_args()

    create_dataset(**vars(args))