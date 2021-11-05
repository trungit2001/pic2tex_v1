import os
import sys
import yaml
import torch
import argparse
import logging
import numpy as np
import pandas.io.clipboard as clipboard

from PIL import Image
from munch import Munch
from timm.models.resnetv2 import ResNetV2
from dataset.dataset import test_transform
from timm.models.layers import StdConv2dSame
from transformers import PreTrainedTokenizerFast

from utils import *
from models import get_model

last_pic = None


def minmax_size(img, max_dimensions=None, min_dimensions=None):
    if max_dimensions is not None:
        ratios = [a/b for a, b in zip(img.size, max_dimensions)]
        if any([r > 1 for r in ratios]):
            size = np.array(img.size)//max(ratios)
            img = img.resize(size.astype(int), Image.BILINEAR)
    if min_dimensions is not None:
        if any([s < min_dimensions[i] for i, s in enumerate(img.size)]):
            padded_im = Image.new('L', min_dimensions, 255)
            padded_im.paste(img, img.getbbox())
            img = padded_im
    return img


def initialize(arguments=None):
    if arguments is None:
        arguments = Munch({'config': 'settings/config.yaml', 'checkpoint': 'checkpoints/weights.pth', 'no_cuda': True, 'no_resize': False})
    logging.getLogger().setLevel(logging.FATAL)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    with open(arguments.config, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    args = parse_args(Munch(params))

    # Error max and min dimensions
    args.max_dimensions = [args.max_width, args.max_height]
    args.min_dimensions = [args.get('min_width', 32), args.get('min_height', 32)]

    args.update(**vars(arguments))
    args.wandb = False
    args.device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'

    model = get_model(args)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))

    if 'image_resizer.pth' in os.listdir(os.path.dirname(args.checkpoint)) and not arguments.no_resize:
        image_resizer = ResNetV2(layers=[2, 3, 3], num_classes=22, global_pool='avg', in_chans=1, drop_rate=.05,
                                 preact=True, stem_type='same', conv_layer=StdConv2dSame).to(args.device)
        image_resizer.load_state_dict(torch.load(os.path.join(os.path.dirname(args.checkpoint), 'image_resizer.pth'), map_location=args.device))
        image_resizer.eval()
    else:
        image_resizer = None
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer)
    return args, model, image_resizer, tokenizer


def call_model(args, model, image_resizer, tokenizer, img=None):
    global last_pic
    encoder, decoder = model.encoder, model.decoder
    if type(img) is bool:
        img = None
    if img is None:
        if last_pic is None:
            print('Provide an image.')
            return
        else:
            img = last_pic.copy()
    else:
        last_pic = img.copy()
    img = minmax_size(pad(img), args.max_dimensions, args.min_dimensions)
    if image_resizer is not None and not args.no_resize:
        with torch.no_grad():
            input_image = pad(img).convert('RGB').copy()
            r, w = 1, img.size[0]
            for i in range(10):
                img = minmax_size(input_image.resize((w, int(input_image.size[1]*r)), Image.BILINEAR if r > 1 else Image.LANCZOS), args.max_dimensions, args.min_dimensions)
                t = test_transform(image=np.array(pad(img).convert('RGB')))['image'][:1].unsqueeze(0)
                w = image_resizer(t.to(args.device)).argmax(-1).item()*32
                if (w/img.size[0] == 1):
                    break
                r *= w/img.size[0]
    else:
        img = np.array(pad(img).convert('RGB'))
        t = test_transform(image=img)['image'][:1].unsqueeze(0)
    im = t.to(args.device)

    with torch.no_grad():
        model.eval()
        device = args.device
        encoded = encoder(im.to(device))
        dec = decoder.generate(torch.LongTensor([args.bos_token])[:, None].to(device), args.max_seq_len,
                               eos_token=args.eos_token, context=encoded.detach(), temperature=args.get('temperature', .25))
        pred = post_process(token2str(dec, tokenizer)[0])
    clipboard.copy(pred)
    return pred


def get_predict(path_img):
    parser = argparse.ArgumentParser(description='Use model', add_help=False)
    parser.add_argument('-t', '--temperature', type=float, default=.333, help='Softmax sampling frequency')
    parser.add_argument('-c', '--config', type=str, default='settings/config.yaml')
    parser.add_argument('-m', '--checkpoint', type=str, default='checkpoints/weights.pth')
    parser.add_argument('-s', '--show', action='store_true', help='Show the rendered predicted latex code')
    parser.add_argument('-f', '--file', type=str, default='', help='Predict LaTeX code from image file instead of clipboard')
    parser.add_argument('-k', '--katex', action='store_true', help='Render the latex code in the browser')
    parser.add_argument('--no-cuda', action='store_true', help='Compute on CPU')
    parser.add_argument('--no-resize', action='store_true', help='Resize the image beforehand')
    arguments, _ = parser.parse_known_args()

    args, *objs = initialize(arguments)

    if os.path.isfile(path_img):
        args.file = path_img
    
    if args.file:
        img = Image.open(args.file)

    pred = call_model(args, *objs, img=img)
    return pred
