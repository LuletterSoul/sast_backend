# %%
import os

from config import Config

model_dir = os.getcwd() + '/models/'

from .utils import *
from torch.autograd import Variable
from torch import optim
import torchvision
from torchvision import transforms
from .network import *
import argparse
import time

parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str, default=Config.CONTENT_DIRECTORY,
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str, default=Config.STYLIZATION_DIRECTORY,
                    help='Directory path to a batch of style images')
parser.add_argument('--content_mask_dir', type=str, default='images/content_masks',
                    help='Directory path to a batch of content masks')
parser.add_argument('--style_mask_dir', type=str, default='images/style_masks',
                    help='Directory path to a batch of style masks')
parser.add_argument('--max_iter', type=int, default=500,
                    help='Max iterations of optimization for low resolution image')
parser.add_argument('--max_iter_hr', type=int, default=200,
                    help='Max iterations of optimization for high resolution image')
parser.add_argument('--update_step', type=int, default=50,
                    help='Update step of loss function and laplacian graph')
parser.add_argument('--update_step_hr', type=int, default=50,
                    help='Update step of loss function and laplacian graph')
parser.add_argument('--img_size', type=int, default=256,
                    help='Image size of low resolution')
parser.add_argument('--img_size_hr', type=int, default=512,
                    help='Image size of high resolution')
parser.add_argument('--kl', type=int, default=50,
                    help='K neighborhoods selection for laplacian graph')
parser.add_argument('--km', type=int, default=1,
                    help='K neighborhoods selection for mutex graph')
# parser.add_argument('--sigma', type=int, default=10,
#                     help='Weight of Variance loss ')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--use_mask', type=bool, default=False)
parser.add_argument('--lw', type=float, default=200)
parser.add_argument('--cw', type=float, default=200)
parser.add_argument('--sw', type=float, default=1)
parser.add_argument('--content_src', type=str, default='datasets/04191521_1000_100_1/warp')
parser.add_argument('--content_list', type=str, default=None)
parser.add_argument('--mean', default='mean', type=str)
# training options0
parser.add_argument('--save_dir',
                    default='output',
                    help='Directory to save the model')

parser.add_argument('--gbp',
                    action='store_true',
                    help='Group by person')

parser.add_argument('--opt_pro',
                    default='process',
                    help='Directory to save the model')

args = parser.parse_args()

save_dir = Path(args.save_dir)
save_dir.mkdir(exist_ok=True, parents=True)

process_dir = args.opt_pro / save_dir
process_dir.mkdir(exist_ok=True, parents=True)

style_weights = [args.sw] * 5
style_layers = ['r11', 'r21', 'r31', 'r41', 'r51']

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
                           transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to BGR
                           transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],  # subtract imagenet mean
                                                std=[1, 1, 1]),
                           transforms.Lambda(lambda x: x.mul_(255)),
                           ])

prep_hr = transforms.Compose([transforms.Resize(args.img_size_hr),
                              transforms.ToTensor(),
                              transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to BGR
                              transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],
                                                   # subtract imagenet mean
                                                   std=[1, 1, 1]),
                              transforms.Lambda(lambda x: x.mul_(255)),
                              ])
postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1. / 255)),
                             transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961],  # add imagenet mean
                                                  std=[1, 1, 1]),
                             transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to RGB
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


def get_prefix(image_name):
    """
    get image name's prefix
    :param image_name:
    :return:
    """
    return os.path.splitext(os.path.basename(image_name))[0]


def compose_prefix(img_name1, img_name2):
    """
    compose stylized image id according content id and style id.
    :param img_name1:
    :param img_name2:
    :return:
    """
    return f'{get_prefix(img_name1)}-{img_name2}.png'.replace(' ', '_')


def st_content_to_style(content_id, style_id, category):
    # get network
    device = torch.device('cuda')
    vgg = VGG()
    vgg.load_state_dict(torch.load(model_dir + 'vgg_conv.pth'))
    for param in vgg.parameters():
        param.requires_grad = False
    if torch.cuda.is_available():
        vgg.cuda()



    outputs = []
    style_images = []
    epoch = 1

    total_time = 0
    avg_time = 0

    content_path = os.path.join(Config.CONTENT_DIRECTORY, content_id)
    style_path = os.path.join(Config.STYLE_DIRECTORY, style_id)

    # load content image tensor
    content_image = load_img(content_path)
    content_image_hr = load_img_hr(content_path)

    # load style image tensor
    style_image = load_img(style_path)
    style_image_hr = load_img_hr(style_path)

    output_path = os.path.join(Config.STYLIZATION_DIRECTORY, compose_prefix(content_id, style_id))

    start = time.time()

    # convert to CUDA tensor
    content_image = content_image.to(device)
    content_image_hr = content_image_hr.to(device)

    style_image = style_image.to(device)
    style_image_hr = style_image_hr.to(device)

    content_mask = None
    style_mask = None

    opt_img = Variable(content_image.data.clone(), requires_grad=True)

    M = Maintainer(vgg, content_image, style_image, content_layers, style_layers, laplacia_layers,
                   device, args.kl,
                   args.km, content_mask, style_mask, args.use_mask, args.mean)

    show_iter = 50
    optimizer = optim.LBFGS([opt_img])
    n_iter = [0]
    while n_iter[0] <= args.max_iter:
        def closure():
            optimizer.zero_grad()
            out = vgg(opt_img, loss_layers)
            # M.add_mutex_constrain(out[-len(mutex_layers):])
            layer_losses = [weights[a] * M.loss_fns[a](A, M.targets[a]) for a, A in enumerate(out)]
            torch.cuda.empty_cache()
            loss = sum(layer_losses)
            loss.backward()
            n_iter[0] += 1
            # print loss
            if n_iter[0] % show_iter == (show_iter - 1):
                print('Iteration: %d, loss: %f' % (n_iter[0] + 1, loss.item()))
            if n_iter[0] % args.update_step == (args.update_step - 1) and not M.laplacian_updated:
                M.update_loss_fns_with_lg(out[-len(laplacia_layers):], M.laplacian_s_feats, args.kl)
                print('Update: Laplacian graph and Loss functions: %d' % (n_iter[0] + 1))
            return loss

        optimizer.step(closure)

    out_img = postp(opt_img.data[0].cpu().squeeze())

    M = Maintainer(vgg, content_image_hr, style_image_hr, content_layers, style_layers, laplacia_layers,
                   device, args.kl,
                   args.km, content_mask, style_mask, args.use_mask, args.mean)

    # now initialise with upsampled lowres result
    opt_img = prep_hr(out_img).unsqueeze(0)
    opt_img = Variable(opt_img.type_as(content_image_hr.data), requires_grad=True)

    optimizer = optim.LBFGS([opt_img])
    n_iter = [0]
    while n_iter[0] <= args.max_iter_hr:

        def closure():
            optimizer.zero_grad()
            out = vgg(opt_img, loss_layers)
            layer_losses = [weights[a] * M.loss_fns[a](A, M.targets[a]) for a, A in enumerate(out)]
            loss = sum(layer_losses)
            torch.cuda.empty_cache()
            loss.backward()

            n_iter[0] += 1
            # print loss
            if n_iter[0] % show_iter == (show_iter - 1):
                print('Iteration: %d, loss: %f' % (n_iter[0] + 1, loss.item()))
            if n_iter[0] % args.update_step_hr == (args.update_step_hr - 1) and not M.laplacian_updated:
                M.update_loss_fns_with_lg(out[-len(laplacia_layers):], M.laplacian_s_feats, args.kl)
                print('Update: Laplacian graph and Loss functions: %d' % (n_iter[0] + 1))
            return loss

        optimizer.step(closure)

    end = time.time()
    total_time += end - start
    avg_time = total_time / epoch
    print(f'Avg time {round(avg_time, 2)}')

    # display result
    out_img_hr = post_tensor(opt_img.data.cpu().squeeze()).unsqueeze(0)
    style_image_hr = post_tensor(style_image_hr.data.cpu().squeeze()).unsqueeze(0)
    style_images.append(style_image_hr)
    outputs.append(out_img_hr)

    torchvision.utils.save_image(out_img_hr.clone(), output_path)
    epoch += 1
    print(f'[SAST]: [{content_id}-{style_id}] style transfer completely')
