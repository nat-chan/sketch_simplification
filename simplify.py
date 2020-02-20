import torch
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.serialization import load_lua
from tqdm import tqdm
from os import path
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description='Sketch simplification demo.')
parser.add_argument('--model', type=str, default='model_gan.t7', help='Model to use.')
parser.add_argument('--img',   type=str, default='test.png',     help='Input image file.')
parser.add_argument('--out',   type=str, default='out.png',      help='File to output.')
opt = parser.parse_args()

use_cuda = torch.cuda.device_count() > 0

cache  = load_lua( opt.model )
model  = cache.model
immean = cache.mean
imstd  = cache.std
model.evaluate()

def preprocess_image(filename):
    data  = Image.open( filename ).convert('L')
    w, h  = data.size[0], data.size[1]
    pw    = 8-(w%8) if w%8!=0 else 0
    ph    = 8-(h%8) if h%8!=0 else 0
    data  = ((transforms.ToTensor()(data)-immean)/imstd).unsqueeze(0)
    if pw!=0 or ph!=0:
       data = torch.nn.ReplicationPad2d( (0,pw,0,ph) )( data ).data
    return data

if __name__ == '__main__':
    batch_size = 32
    root = "/home/natsuki/danbooru2019"
    with open(path.join(root, 'list'), 'r') as f:
        lines = f.read().splitlines()
    for j in tqdm(range(len(lines) // batch_size)):
        data = torch.cat([
            preprocess_image(path.join(root, 'sketchKeras_pured', lines[j*batch_size+i]))
            for i in range(batch_size)
        ])
        if use_cuda:
           pred = model.cuda().forward( data.cuda() ).float()
        else:
           pred = model.forward( data )
        for i in range(batch_size):
            save_image(pred[i], path.join(root, 'sim_pured', lines[j*batch_size+i]))
