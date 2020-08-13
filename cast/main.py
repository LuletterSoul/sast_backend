# %%

# %pylab inline
import os

image_dir = os.getcwd() + '/images/'
model_dir = os.getcwd() + '/models/'

from .transforms import *
from torch.autograd import Variable
from torch import optim
import torchvision
from torchvision import transforms
from .network import *
import argparse
import time

parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str, default='images/content',
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str, default='images/style',
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


content_dataset = FlatFolderDataset(args.content_dir, args.content_mask_dir, prep, prep_hr, mask_tf)
style_dataset = FlatFolderDataset(args.style_dir, args.style_mask_dir, prep, prep_hr, mask_tf)
# content_hr_dataset = FlatFolderDataset(args.content_dir, prep)
# style_hr_dataset = FlatFolderDataset(args.style_dir, prep)

content_loader = data.DataLoader(
    content_dataset, batch_size=1, shuffle=False,
    num_workers=4)
style_loader = data.DataLoader(
    style_dataset, batch_size=1, shuffle=False,
    num_workers=4)

# content_hr_loader = data.DataLoader(
#     content_hr_dataset, batch_size=1, shuffle=False,
#     num_workers=0)
# style_hr_loader = data.DataLoader(
#     style_hr_dataset, batch_size=1, shuffle=False,
#     num_workers=0)

# # print(len(content_loader))
# print(len(style_loader))
# style_dataset content_iter = iter()
# style_iter = iter()
# %%

# get network
device = torch.device('cuda')
vgg = VGG()
vgg.load_state_dict(torch.load(model_dir + 'vgg_conv.pth'))
for param in vgg.parameters():
    param.requires_grad = False
if torch.cuda.is_available():
    vgg.cuda()

# %%

# load images, ordered as [style_image, content_image]
# img_dirs = [image_dir, image_dir]
# img_names = ['vangogh_starry_night.jpg', 'Tuebingen_Neckarfront.jpg']
# imgs = [Image.open(img_dirs[i] + name) for i, name in enumerate(img_names)]
# imgs_torch = [prep(img) for img in imgs]
# if torch.cuda.is_available():
#     imgs_torch = [Variable(img.unsqueeze(0).cuda()) for img in imgs_torch]
# else:
#     imgs_torch = [Variable(img.unsqueeze(0)) for img in imgs_torch]

outputs = []
style_images = []
epoch = 1

total_time = 0
avg_time = 0

for content_image, content_image_hr, content_mask, content_name in content_loader:
    # print(content_name)
    for style_image, style_image_hr, style_mask, style_name in style_loader:

        if not args.gbp:
            output_path = os.path.join(args.save_dir, f'{content_name[0]}-{style_name[0]}.png')
        else:
            person_name, extention = os.path.splitext(content_name[0])
            output_dir = os.path.join(args.save_dir, person_name).replace(' ', '_')
            if not os.path.exists(output_dir):
                Path(output_dir).mkdir(exist_ok=True, parents=True)
            output_path = os.path.join(output_dir, f'{content_name[0]}-{style_name[0]}.png').replace(' ', '_')

        if os.path.exists(output_path):
            print(f'Stylization exist in {output_path}')
            continue
        start = time.time()
        content_image = content_image.to(device)
        content_image_hr = content_image_hr.to(device)

        content_mask = content_mask.to(device)
        style_mask = style_mask.to(device)

        style_image = style_image.to(device)
        style_image_hr = style_image_hr.to(device)
        # style_image = style_image.squeeze(0)
        # content_image = content_image.squeeze(0)
        # style_image, content_image = imgs_torch
        # opt_img = Variable(torch.randn(content_image.size()).type_as(content_image.data), requires_grad=True) #random init
        opt_img = Variable(content_image.data.clone(), requires_grad=True)
        # %% # display images
        # for img in imgs:
        #     imshow(img).show()

        # %%

        # define layers, loss functions, weights and compute optimization targets

        # style_feats = vgg(style_image, style_layers)
        # content_feats = vgg(content_image, content_layers)
        # # content_complete_feats = vgg(content_image, style_layers)
        # laplacian_c_feats = vgg(content_image, laplacia_layers)P00004.jpg
        # laplacian_s_feats = vgg(style_image, laplacia_layers)
        #
        # # init Laplacian graph
        # laplacian_graphs = [
        #     cal_laplacian_graph(laplacian_c_feats[idx], laplacian_s_feats[idx], laplacian_s_feats[idx], 7)
        #     for idx in
        #     range(len(laplacian_s_feats))]
        # # compute optimization targets
        #
        # loss_fns = [GramMSELoss()] * len(style_layers) + [nn.MSELoss()] * len(content_layers) + [ConsistencyLoss(l) for
        #                                                                                          l in laplacian_graphs]
        # if torch.cuda.is_available():
        #     loss_fns = [loss_fn.cuda() for loss_fn in loss_fns]
        #
        # style_targets = [GramMatrix()(A).detach() for A in style_feats]
        # content_targets = [A.detach() for A in content_feats]
        # laplacia_targets = [A.detach() for A in laplacian_s_feats]
        # targets = style_targets + content_targets + laplacia_targets
        print(f'content_image_size: {content_image.size()}')
        print(f'style_image_size: {style_image.size()}')
        M = Maintainer(vgg, content_image, style_image, content_layers, style_layers, laplacia_layers,
                       device, args.kl,
                       args.km, content_mask, style_mask, args.use_mask, args.mean)

        # %%

        # run style transfer
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
                # tmp_out_img = post_tensor(opt_img.data.cpu().squeeze()).unsqueeze(0)
                # torchvision.utils.save_image(tmp_out_img, os.path.join(str(process_dir),
                #                                                        f'{content_name[0]}-{style_name[0]}-{n_iter[0]}_lr.png'))
                n_iter[0] += 1
                # print loss
                if n_iter[0] % show_iter == (show_iter - 1):
                    print('Iteration: %d, loss: %f' % (n_iter[0] + 1, loss.item()))
                if n_iter[0] % args.update_step == (args.update_step - 1) and not M.laplacian_updated:
                    # pass
                    # Using output as content image to update laplacian graph dynamiclly during trainig.
                    # M.update_loss_fns_with_lg(out[-len(laplacia_layers) + -len(mutex_layers):-len(mutex_layers)],
                    #                           M.laplacian_s_feats)
                    M.update_loss_fns_with_lg(out[-len(laplacia_layers):], M.laplacian_s_feats, args.kl)
                    # M.laplacian_updated = True
                    # M.update_loss_fns_with_lg(out[len(content_layers) + len(style_layers):], M.laplacian_s_feats)
                    print('Update: Laplacian graph and Loss functions: %d' % (n_iter[0] + 1))
                    # print('Update laplacian graph and loss functions: %d' % (n_iter[0] + 1))
                #             print([loss_layers[li] + ': ' +  str(l.data[0]) for li,l in enumerate(layer_losses)]) #loss of each layer
                return loss


            optimizer.step(closure)

        # display result
        out_img = postp(opt_img.data[0].cpu().squeeze())
        # imshow(out_img)
        # gcf().set_size_inches(10,10)
        # torchvision.utils.save_image('./output.png', opt_img)

        # %%

        # make the image high-resolution as described in
        # "Controlling Perceptual Factors in Neural Style Transfer", Gatys et al.
        # (https://arxiv.org/abs/1611.07865)

        # hr preprocessing

        # prep hr images
        # imgs_torch = [prep_hr(img) for img in imgs]
        # if torch.cuda.is_available():
        #     imgs_torch = [Variable(img.unsqueeze(0).cuda()) for img in imgs_torch]
        # else:
        #     imgs_torch = [Variable(img.unsqueeze(0)) for img in imgs_torch]
        # style_image, content_image = imgs_torch

        # Update Global Training Components

        M = Maintainer(vgg, content_image_hr, style_image_hr, content_layers, style_layers, laplacia_layers,
                       device, args.kl,
                       args.km, content_mask, style_mask, args.use_mask, args.mean)

        # now initialise with upsampled lowres result
        opt_img = prep_hr(out_img).unsqueeze(0)
        opt_img = Variable(opt_img.type_as(content_image_hr.data), requires_grad=True)

        # style_targets = [GramMatrix()(A).detach() for A in vgg(style_image_hr, style_layers)]
        # content_targets = [A.detach() for A in vgg(content_image_hr, content_layers)]
        # laplacia_targets = [GramMatrix()(A).detach() for A in vgg(style_image_hr, style_layers)]
        # targets = style_targets + content_targets + laplacia_targets

        # %%

        # run style transfer for high res
        optimizer = optim.LBFGS([opt_img])
        n_iter = [0]
        while n_iter[0] <= args.max_iter_hr:

            def closure():
                optimizer.zero_grad()
                out = vgg(opt_img, loss_layers)
                # M.add_mutex_constrain(out[-len(mutex_layers):])
                layer_losses = [weights[a] * M.loss_fns[a](A, M.targets[a]) for a, A in enumerate(out)]
                loss = sum(layer_losses)
                # loss = sum(layer_losses)
                torch.cuda.empty_cache()
                loss.backward()

                # tmp_out_img = post_tensor(opt_img.data.cpu().squeeze()).unsqueeze(0)
                # torchvision.utils.save_image(tmp_out_img, os.path.join(str(process_dir),
                #                                                        f'{content_name[0]}-{style_name[0]}-{n_iter[0]}_hr.png'))
                n_iter[0] += 1
                # print loss
                if n_iter[0] % show_iter == (show_iter - 1):
                    # print('Iteration: %d, loss: %f' % (n_iter[0] + 1, loss.data[0]))
                    print('Iteration: %d, loss: %f' % (n_iter[0] + 1, loss.item()))
                if n_iter[0] % args.update_step_hr == (args.update_step_hr - 1) and not M.laplacian_updated:
                    # Using output as content image to update laplacian graph dynamiclly during trainig.
                    M.update_loss_fns_with_lg(out[-len(laplacia_layers):], M.laplacian_s_feats, args.kl)
                    # M.update_loss_fns_with_lg(out[-len(laplacia_layers) + -len(mutex_layers):-len(mutex_layers)],
                    #                           M.laplacian_s_feats)
                    # pass
                    # M.laplacian_updated = True
                    # M.update_loss_fns_with_lg(out[len(content_layers) + len(style_layers):], M.laplacian_s_feats)
                    print('Update: Laplacian graph and Loss functions: %d' % (n_iter[0] + 1))
                # k          print([loss_layers[li] + ': ' +  str(l.data[0]) for li,l in enumerate(layer_losses)]) #loss of each layer
                return loss


            optimizer.step(closure)

        end = time.time()
        total_time += end - start
        avg_time = total_time / epoch
        print(f'Avg time {round(avg_time, 2)}')
        # display result
        out_img_hr = post_tensor(opt_img.data.cpu().squeeze()).unsqueeze(0)
        style_image_hr = post_tensor(style_image_hr.data.cpu().squeeze()).unsqueeze(0)
        # imshow(out_img_hr)
        style_images.append(style_image_hr)
        outputs.append(out_img_hr)

        torchvision.utils.save_image(out_img_hr.clone(), output_path)

        if (epoch + 1) % args.batch_size == 0:
            style_images = torch.cat(style_images, dim=0)
            outputs = torch.cat(outputs, dim=0)
            o = torch.cat([style_images, outputs], dim=0)
            path = os.path.join(args.save_dir,
                                'total-{}-{}-{}.png'.format(epoch, content_name[0], style_name[0]))
            torchvision.utils.save_image(o, path, nrow=args.batch_size)
            print('Save to [{}]'.format(path))
            outputs = []
            style_images = []
        print('Done: [{}-{}].'.format(content_name[0], style_name[0]))
        epoch += 1
