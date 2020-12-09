#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: cast.py
@time: 2020/8/16 22:15
@version 1.0
@desc:
"""

# %%
import time
from cast.network import *
from torchvision import transforms
import torchvision
from torch import optim
from torch.autograd import Variable
from cast.utils import *
from workers.stream import ManagedModel, mp, threading, run_redis_workers_forever
from sockets import *
import os
import shutil
import traceback

import cv2

from cast.cast import warp_content_to_style_images
from utils.utils import compose_prefix_id, construct_cast_msg, get_prefix, compose_prefix, parse_devices

model_dir = os.getcwd() + '/models/'


model_dir = os.getcwd() + '/models/'


# parser = argparse.ArgumentParser()
# # Basic options
# parser.add_argument('--content_dir', type=str, default=Config.CONTENT_DIRECTORY,
#                     help='Directory path to a batch of content images')
# parser.add_argument('--style_dir', type=str, default=Config.STYLIZATION_DIRECTORY,
#                     help='Directory path to a batch of style images')
# parser.add_argument('--content_mask_dir', type=str, default='images/content_masks',
#                     help='Directory path to a batch of content masks')
# parser.add_argument('--style_mask_dir', type=str, default='images/style_masks',
#                     help='Directory path to a batch of style masks')
# parser.add_argument('--max_iter', type=int, default=500,
#                     help='Max iterations of optimization for low resolution image')
# parser.add_argument('--max_iter_hr', type=int, default=200,
#                     help='Max iterations of optimization for high resolution image')
# parser.add_argument('--update_step', type=int, default=50,
#                     help='Update step of loss function and laplacian graph')
# parser.add_argument('--update_step_hr', type=int, default=50,
#                     help='Update step of loss function and laplacian graph')
# parser.add_argument('--img_size', type=int, default=256,
#                     help='Image size of low resolution')
# parser.add_argument('--img_size_hr', type=int, default=512,
#                     help='Image size of high resolution')
# parser.add_argument('--kl', type=int, default=50,
#                     help='K neighborhoods selection for laplacian graph')
# parser.add_argument('--km', type=int, default=1,
#                     help='K neighborhoods selection for mutex graph')
# # parser.add_argument('--sigma', type=int, default=10,
# #                     help='Weight of Variance loss ')
# parser.add_argument('--batch_size', type=int, default=4)
# parser.add_argument('--use_mask', type=bool, default=False)
# parser.add_argument('--lw', type=float, default=200)
# parser.add_argument('--cw', type=float, default=200)
# parser.add_argument('--sw', type=float, default=1)
# parser.add_argument('--content_src', type=str, default='datasets/04191521_1000_100_1/warp')
# parser.add_argument('--content_list', type=str, default=None)
# parser.add_argument('--mean', default='mean', type=str)
# # training options0
# parser.add_argument('--save_dir',
#                     default='output',
#                     help='Directory to save the model')
#
# parser.add_argument('--gbp',
#                     action='store_true',
#                     help='Group by person')
#
# parser.add_argument('--opt_pro',
#                     default='process',
#                     help='Directory to save the model')

# args = parser.parse_args()
class DefaultCastConfig:
    content_dir = Config.CONTENT_DIRECTORY
    style_dir = Config.STYLE_DIRECTORY
    save_dir = 'output'
    content_mask_dir = 'images/content_masks'
    style_mask_dir = 'images/style_masks'
    # max_iter = 200
    # max_iter_hr = 100
    max_iter = 200
    max_iter_hr = 100
    update_step = 30
    update_step_hr = 50
    img_size = 256
    img_size_hr = 512
    kl = 1
    km = 1
    sigma = 10
    batch_size = 4
    use_mask = False
    lw = 30
    cw = 1
    sw = 1
    content_src = ''
    content_list = []
    mean = 'mean'
    gbp = False
    opt_pro = 'process'


args = DefaultCastConfig()

save_dir = Path(args.save_dir)
save_dir.mkdir(exist_ok=True, parents=True)

process_dir = args.opt_pro / save_dir
process_dir.mkdir(exist_ok=True, parents=True)

# style_weights = [args.sw] * 5
# style_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
style_weights = []
style_layers = []

alpha = args.cw / (args.cw + args.lw)
beta = args.lw / (args.cw + args.lw)

content_layers = ['r42']
content_weights = [alpha]

laplacia_layers = ['r32']
laplacia_weights = [beta]

mutex_layers = []
mutex_weights = []

loss_layers = style_layers + content_layers + laplacia_layers + mutex_layers
weights = style_weights + content_weights + laplacia_weights + mutex_weights
prep = transforms.Compose([transforms.Resize(args.img_size),
                           transforms.ToTensor(),
                           transforms.Lambda(
                               lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to BGR
                           transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],  # subtract imagenet mean
                                                std=[1, 1, 1]),
                           transforms.Lambda(lambda x: x.mul_(255)),
                           ])

prep_hr = transforms.Compose([transforms.Resize(args.img_size_hr),
                              transforms.ToTensor(),
                              transforms.Lambda(
                                  lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to BGR
                              transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],
                                                   # subtract imagenet mean
                                                   std=[1, 1, 1]),
                              transforms.Lambda(lambda x: x.mul_(255)),
                              ])
postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1. / 255)),
                             transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961],  # add imagenet mean
                                                  std=[1, 1, 1]),
                             transforms.Lambda(
                                 lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to RGB
                             ])
mask_tf = transforms.Compose([ToUnNormalizedTensor()])
postpb = transforms.Compose([transforms.ToPILImage()])


def postp(tensor):  # to clip results in the range [0,1]
    t = post_tensor(tensor)
    img = postpb(t)
    return img
    # return t


def post_tensor(tensor):
    t = postpa(tensor)
    t[t > 1] = 1
    t[t < 0] = 0
    return t


def load_img(path):
    img = Image.open(path).convert('RGB')
    return prep(img)


def load_img_hr(path):
    img = Image.open(path).convert('RGB')
    return prep_hr(img)


class CastModel(ManagedModel):

    def __init__(self, gpu_id=None):
        super().__init__(gpu_id)

    def init_model(self, *args, **kwargs):
        print(f'[Cast]: # init from CUDA device')
        print(f'[Cast]: Loading models...')
        self.device = torch.device('cuda')
        self.vgg = VGG()
        self.vgg.load_state_dict(torch.load(os.path.join(
            Config.CAST_WORK_DIR, 'models/vgg_conv.pth')))
        for param in self.vgg.parameters():
            param.requires_grad = False
        if torch.cuda.is_available():
            self.vgg.cuda()
        print(f'[Cast]: Load models completely!')

    def predict(self, msg):
        try:
            content_id = msg.get('content_id')
            style_id = msg.get('style_id')
            category = msg.get('category')
            style_category = msg.get('style_category')
            content_category = msg.get('content_category')


            # warped image will be saved in content directory as input of stylization
            warped_id = warp_content_to_style_images(
                content_id, style_id, category=content_category)

            # related image output directory
            style_path = os.path.join(Config.STYLE_DIRECTORY, style_category, style_id)
            content_path = os.path.join(Config.CONTENT_DIRECTORY, warped_id)
            stylization_id = compose_prefix_id(content_id, style_id)
            output_path = os.path.join(
                Config.STYLIZATION_DIRECTORY, stylization_id)
            intermediate_id_prefix = get_prefix(warped_id)

            # create intermediate stylized image output directory.
            intermediate_output_dir = os.path.join(
                Config.STYLIZATION_DIRECTORY, intermediate_id_prefix)
            os.makedirs(intermediate_output_dir, exist_ok=True)
            self.render(msg, content_path, style_path, intermediate_id_prefix, intermediate_output_dir, stylization_id,
                            output_path)
        except Exception as e:
            msg['status'] = 'failed'
            synthesis_failed(msg)
            traceback.print_exc()
        finally:
            shutil.rmtree(intermediate_output_dir)
            # we don't need warped image saved at content directory.
            if os.path.exists(content_path):
                os.remove(content_path)

    def render(self, msg, content_path, style_path, intermediate_id_prefix, intermediate_output_dir, stylization_id,
               output_path):
        epoch = 1
        total_time = 0
        # avg_time = 0
        # original_content_id = content_id
        # content_id = warped_id
        width = msg['width']
        height = msg['height']

        # load content image tensor
        content_image = load_img(content_path)
        content_image_hr = load_img_hr(content_path)
        # load style image tensor
        style_image = load_img(style_path)
        style_image_hr = load_img_hr(style_path)
        msg['stylization_id'] = stylization_id

        start = time.time()
        # convert to CUDA tensor
        content_image = content_image.to(self.device).unsqueeze(0)
        content_image_hr = content_image_hr.to(self.device).unsqueeze(0)
        style_image = style_image.to(self.device).unsqueeze(0)
        style_image_hr = style_image_hr.to(self.device).unsqueeze(0)
        total_update_steps = args.max_iter + args.max_iter_hr
        content_mask = None
        style_mask = None
        opt_img = Variable(content_image.data.clone(), requires_grad=True)
        M = Maintainer(self.vgg, content_image, style_image, content_layers, style_layers, laplacia_layers,
                       self.device, args.kl,
                       args.km, content_mask, style_mask, args.use_mask, args.mean)
        show_iter = 50
        optimizer = optim.LBFGS([opt_img])
        n_iter = [0]
        while n_iter[0] <= args.max_iter:
            def closure():
                optimizer.zero_grad()
                out = self.vgg(opt_img, loss_layers)
                # M.add_mutex_constrain(out[-len(mutex_layers):])
                layer_losses = [weights[a] * M.loss_fns[a]
                                (A, M.targets[a]) for a, A in enumerate(out)]
                torch.cuda.empty_cache()
                loss = sum(layer_losses)
                loss.backward()
                n_iter[0] += 1

                # print loss
                if n_iter[0] % show_iter == (show_iter - 1):
                    print('Iteration: %d, loss: %f' %
                          (n_iter[0] + 1, loss.item()))
                if n_iter[0] % args.update_step == (args.update_step - 1) and not M.laplacian_updated:
                    M.update_loss_fns_with_lg(
                        out[-len(laplacia_layers):], M.laplacian_s_feats, args.kl)
                    print('Update: Laplacian graph and Loss functions: %d' %
                          (n_iter[0] + 1))

                return loss

            # each intermediate stylized image will be
            # saved into a directory in case truncated image read by client end.
            intermediate_id = f'{intermediate_id_prefix}_{n_iter[0]}.png'
            warped_output_path = os.path.join(
                intermediate_output_dir, intermediate_id)
            self.save_optimized_img(
                opt_img, warped_output_path, height=height, width=width)
            # notify client end with current progress
            current_cost_time = round(time.time() - start, 2)
            body = construct_cast_msg(msg, intermediate_id, n_iter[0], current_cost_time, total_update_steps,
                                      intermediate_id_prefix)
            synthesising(body, notify_fetch=True)
            optimizer.step(closure)
        out_img = postp(opt_img.data[0].cpu().squeeze())

        del opt_img

        M = Maintainer(self.vgg, content_image_hr, style_image_hr, content_layers, style_layers, laplacia_layers,
                       self.device, args.kl,
                       args.km, content_mask, style_mask, args.use_mask, args.mean)
        # now initialise with upsampled lowres result
        opt_img = prep_hr(out_img).unsqueeze(0)
        opt_img = Variable(opt_img.type_as(
            content_image_hr.data), requires_grad=True)
        optimizer = optim.LBFGS([opt_img])
        n_iter = [0]
        while n_iter[0] <= args.max_iter_hr:
            def closure():
                optimizer.zero_grad()
                out = self.vgg(opt_img, loss_layers)
                layer_losses = [weights[a] * M.loss_fns[a]
                                (A, M.targets[a]) for a, A in enumerate(out)]
                loss = sum(layer_losses)
                torch.cuda.empty_cache()
                loss.backward()

                n_iter[0] += 1
                # print loss
                if n_iter[0] % show_iter == (show_iter - 1):
                    print('Iteration: %d, loss: %f' %
                          (n_iter[0] + 1, loss.item()))
                if n_iter[0] % args.update_step_hr == (args.update_step_hr - 1) and not M.laplacian_updated:
                    M.update_loss_fns_with_lg(
                        out[-len(laplacia_layers):], M.laplacian_s_feats, args.kl)
                    print('Update: Laplacian graph and Loss functions: %d' %
                          (n_iter[0] + 1))

                return loss

            intermediate_id = f'{intermediate_id_prefix}_hr_{n_iter[0]}.png'
            warped_output_path = os.path.join(
                intermediate_output_dir, intermediate_id)
            self.save_optimized_img(opt_img, warped_output_path)
            current_cost_time = round(time.time() - start, 2)
            body = construct_cast_msg(msg, intermediate_id, args.max_iter + n_iter[0], current_cost_time,
                                      total_update_steps, intermediate_id_prefix)
            synthesising(body, notify_fetch=True)
            optimizer.step(closure)
        end = time.time()
        total_time += end - start
        avg_time = total_time / epoch
        print(f'Avg time {round(avg_time, 2)}')
        # display result
        self.save_optimized_img(opt_img, output_path)
        epoch += 1
        print(f'[SAST]: [{stylization_id}] style transfer completely')

        msg['timestamp'] = time.time()
        synthesis_complete(msg)
        # clean intermediate stylized images
        optimizer.zero_grad()
        torch.cuda.empty_cache() 
        del opt_img

    def save_optimized_img(self, opt_img, output_path, height=None, width=None):
        out_img_hr = post_tensor(opt_img.data.cpu().squeeze()).unsqueeze(0)
        torchvision.utils.save_image(out_img_hr.clone(), output_path)
        if height is not None and width is not None:
            img = cv2.imread(output_path)
            img = cv2.resize(img, (width, height),
                             interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(output_path, img)


# parser = argparse.ArgumentParser()
# # Basic options
# parser.add_argument('--content_dir', type=str, default=Config.CONTENT_DIRECTORY,
#                     help='Directory path to a batch of content images')
# parser.add_argument('--style_dir', type=str, default=Config.STYLIZATION_DIRECTORY,
#                     help='Directory path to a batch of style images')
# parser.add_argument('--content_mask_dir', type=str, default='images/content_masks',
#                     help='Directory path to a batch of content masks')
# parser.add_argument('--style_mask_dir', type=str, default='images/style_masks',
#                     help='Directory path to a batch of style masks')
# parser.add_argument('--max_iter', type=int, default=500,
#                     help='Max iterations of optimization for low resolution image')
# parser.add_argument('--max_iter_hr', type=int, default=200,
#                     help='Max iterations of optimization for high resolution image')
# parser.add_argument('--update_step', type=int, default=50,
#                     help='Update step of loss function and laplacian graph')
# parser.add_argument('--update_step_hr', type=int, default=50,
#                     help='Update step of loss function and laplacian graph')
# parser.add_argument('--img_size', type=int, default=256,
#                     help='Image size of low resolution')
# parser.add_argument('--img_size_hr', type=int, default=512,
#                     help='Image size of high resolution')
# parser.add_argument('--kl', type=int, default=50,
#                     help='K neighborhoods selection for laplacian graph')
# parser.add_argument('--km', type=int, default=1,
#                     help='K neighborhoods selection for mutex graph')
# # parser.add_argument('--sigma', type=int, default=10,
# #                     help='Weight of Variance loss ')
# parser.add_argument('--batch_size', type=int, default=4)
# parser.add_argument('--use_mask', type=bool, default=False)
# parser.add_argument('--lw', type=float, default=200)
# parser.add_argument('--cw', type=float, default=200)
# parser.add_argument('--sw', type=float, default=1)
# parser.add_argument('--content_src', type=str, default='datasets/04191521_1000_100_1/warp')
# parser.add_argument('--content_list', type=str, default=None)
# parser.add_argument('--mean', default='mean', type=str)
# # training options0
# parser.add_argument('--save_dir',
#                     default='output',
#                     help='Directory to save the model')
#
# parser.add_argument('--gbp',
# content_id                     action='store_true',
#                     help='Group by person')
#
# parser.add_argument('--opt_pro',
#                     default='process',
#                     help='Directory to save the model')
#
# self.args = parser.parse_args()
#
# save_dir = Path(self.args.save_dir)
# save_dir.mkdir(exist_ok=True, parents=True)
#
# process_dir = self.args.opt_pro / save_dir
# process_dir.mkdir(exist_ok=True, parents=True)
#
# style_weights = [self.args.sw] * 5
# style_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
#
# alpha = self.args.cw / (self.args.cw + self.args.lw)
# beta = self.args.lw / (self.args.cw + self.args.lw)
#
# content_layers = ['r42']
# content_weights = [alpha]
#
# laplacia_layers = ['r32']
# laplacia_weights = [beta]
#
# mutex_layers = []
# mutex_weights = []
#
# loss_layers = style_layers + content_layers + laplacia_layers + mutex_layers
# weights = style_weights + content_weights + laplacia_weights + mutex_weights
# prep = transforms.Compose([transforms.Resize(self.args.img_size),
#                            transforms.ToTensor(),
#                            transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to BGR
#                            transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],
#                                                 # subtract imagenet mean
#                                                 std=[1, 1, 1]),
#                            transforms.Lambda(lambda x: x.mul_(255)),
#                            ])
#
# prep_hr = transforms.Compose([transforms.Resize(self.args.img_size_hr),
#                               transforms.ToTensor(),
#                               transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to BGR
#                               transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],
#                                                    # subtract imagenet mean
#                                                    std=[1, 1, 1]),
#                               transforms.Lambda(lambda x: x.mul_(255)),
#                               ])
# postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1. / 255)),
#                              transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961],
#                                                   # add imagenet mean
#                                                   std=[1, 1, 1]),
#                              transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to RGB
#                              ])
# mask_tf = transforms.Compose([ToUnNormalizedTensor()])
# postpb = transforms.Compose([transforms.ToPILImage()])

def create_cast_worker():
    model_init_args = ()
    destroy_event = mp.Event()
    devices = parse_devices(Config.CAST_DEVICES)
    # batch_size = Config.MAST_BATCH_SIZE
    # worker_num = Config.MAST_WORKER_NUM
    thread = threading.Thread(target=run_redis_workers_forever, args=(CastModel, Config.CAST_BATCH_SIZE, 0.1,
                                                                      Config.CAST_WORKER_NUM, devices,
                                                                      Config.REDIS_BROKER_URL, Config.CAST_CHANNEL,
                                                                      model_init_args, None, destroy_event,),
                              daemon=True)
    thread.start()
    return thread, destroy_event

# %%
