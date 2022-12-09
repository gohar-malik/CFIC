import os
from os.path import join
import json
import argparse
from time import time
from tqdm import tqdm

from predict import Predictor

parser = argparse.ArgumentParser(description='generate_captions')
parser.add_argument('--root_dir', default='/ssd_data/gohar/coco',
                    type=str, help='Root dir where val2014 and annotations dir are located')
parser.add_argument('--model', default='conceptual-captions',
                    type=str, help='Pretrained model used to generate captions [conceptual-captions or coco]')
parser.add_argument('--orig_att', type=str, default='dark')
parser.add_argument('--gpu', type=str, default='0', help='gpu index, -1 for cpu')

args = parser.parse_args()

if __name__ == "__main__":
    results_dir = "../captions"
    os.makedirs(results_dir, exist_ok=True)
    im_dir_list = [ join(args.root_dir, f"stgan/val2014_{args.orig_att}_mod"), 
                    join(args.root_dir, f"stgan/val2014_{args.orig_att}_orig")]

    model = args.model
    device = "cpu" if args.gpu == "-1" else f"cuda:{args.gpu}"
    use_beam_search=True
    predictor = Predictor(device)

    for im_dir in im_dir_list:
        print(im_dir)
        result_path = join(results_dir, f"{im_dir.split('/')[-1]}_{model}.json")

        # inference loop
        captions_dict = {}
        tic = time()
        for i, im_file in enumerate(tqdm(os.listdir(im_dir))):
            name = im_file.split(".")[0]
            # print(name)

            im_path = join(im_dir, im_file)

            captions = predictor.predict(im_path, model, use_beam_search=use_beam_search)

            captions_dict[name] = captions
            # if i >=10:
            #     break

        toc = time()
        print(f"Avg Inference Time for {i+1} images: {(toc-tic)/(i+1):.3f} secs\n")

        # saving captions
        # print(captions_dict)
        with open(result_path, "w") as f:
            json.dump(captions_dict, f)
