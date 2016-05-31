from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import matplotlib.pyplot as plt
import skimage.io as io
import os, sys, getopt

import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')

annFile = './annotations/captions_val2014.json'
coco = COCO(annFile)
results_dir = 'best_model/results'
final_eval_file = results_dir + '/evaluation_val.json'

def evaluateModel(model_json):
    cocoRes = coco.loadRes(model_json)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()  
    cocoEval.evaluate()
    results = {}
    for metric, score in cocoEval.eval.items():
        results[metric] = score
    return results

def main(argv):
    opts, args = getopt.getopt(argv, 'e:')
    for opt, arg in opts:
        if opt == '-e':
            epoch = int(arg)

    resFile = results_dir + ('/val_res_%d.json' % epoch)
    results = evaluateModel(resFile)

    if not os.path.isfile(final_eval_file):
        all_results_json = {}
    else:
        with open(final_eval_file,'r') as f:
            all_results_json = json.load(f)

    all_results_json[epoch] = results
    with open(final_eval_file,'w') as f:
        json.dump(all_results_json, f, sort_keys=True, indent=4)

if __name__ == '__main__':
    main(sys.argv[1:])