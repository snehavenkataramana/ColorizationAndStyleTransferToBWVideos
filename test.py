
import glob
import os
from options.train_options import TrainOptions
from models import create_model
from util.visualizer import save_images
from util import html

import string
import torch
import torchvision
import torchvision.transforms as transforms

from util import util
from IPython import embed
import numpy as np


if __name__ == '__main__':
    #sample_ps = [1., .125, .03125]
    sample_ps = [1.0]
    to_visualize = ['gray', 'hint', 'hint_ab', 'fake_entr', 'real', 'fake_reg', 'real_ab', 'fake_ab_reg', ]
    S = len(sample_ps)

    opt = TrainOptions().parse()
    opt.load_model = True
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.display_id = -1  # no visdom display
    opt.phase = 'val'
    opt.dataroot = './dataset/custom/val/'
    opt.serial_batches = True
    opt.aspect_ratio = 1.

    dataset = torchvision.datasets.ImageFolder(opt.dataroot,
                                               transform=transforms.Compose([
                                                   transforms.Resize((opt.loadSize, opt.loadSize)),
                                                   transforms.ToTensor()]))
    dataset_loader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=not opt.serial_batches)

    ref_dataset = torchvision.datasets.ImageFolder('./dataset/ref/',
                                               transform=transforms.Compose([
                                                   transforms.Resize((opt.loadSize, opt.loadSize)),
                                                   transforms.ToTensor()]))
    ref_dataset_loader = torch.utils.data.DataLoader(
        ref_dataset, batch_size=opt.batch_size, shuffle=not opt.serial_batches)

    model = create_model(opt)
    model.setup(opt)
    model.eval()

    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

    # statistics
    psnrs = np.zeros((opt.how_many, S))
    entrs = np.zeros((opt.how_many, S))

    # for i, data_raw in enumerate(dataset_loader):
    for i, (data_raw, ref_raw) in enumerate(zip(dataset_loader, ref_dataset_loader)):

        # i, data_raw = ds
        # j, ref_raw = ref_ds

        data_raw[0] = data_raw[0].cuda()
        data_raw[0] = util.crop_mult(data_raw[0], mult=8)

        ref_raw[0] = ref_raw[0].cuda()
        ref_raw[0] = util.crop_mult(ref_raw[0], mult=8)

        # with no points
        for (pp, sample_p) in enumerate(sample_ps):
            img_path = ('%08d_%.3f' % (i, sample_p)).replace('.', 'p')

            # data = util.get_colorization_data(data_raw, opt, ab_thresh=0., p=sample_p)
            data = util.get_colorization_data_with_ref(
                data_raw, ref_raw, opt, ab_thresh=0., p=sample_p)

            model.set_input(data)
            model.test(True)  # True means that losses will be computed
            visuals = util.get_subset_dict(model.get_current_visuals(), to_visualize)

            psnrs[i, pp] = util.calculate_psnr_np(util.tensor2im(visuals['real']), util.tensor2im(visuals['fake_reg']))
            entrs[i, pp] = model.get_current_losses()['G_entr']

            save_images(
                i, webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

            print('Saving ', img_path)

        print('Processing image', i)

        if i % 5 == 0:
            print('processing (%04d)-th image... %s' % (i, img_path))
            #if i != 0:
            #    break

        # if i == opt.how_many - 1:
        #     break

    webpage.save()

    # Compute and print some summary statistics
    psnrs_mean = np.mean(psnrs, axis=0)
    psnrs_std = np.std(psnrs, axis=0) / np.sqrt(opt.how_many)

    entrs_mean = np.mean(entrs, axis=0)
    entrs_std = np.std(entrs, axis=0) / np.sqrt(opt.how_many)

    for (pp, sample_p) in enumerate(sample_ps):
        print('p=%.3f: %.2f+/-%.2f' % (sample_p, psnrs_mean[pp], psnrs_std[pp]))


    gif_name = 'out'
    file_list = glob.glob('./results/siggraph_reg2/val_latest/**/*.png')
    valid_files = []
    for img_file in file_list:
        if 'fake_reg' in img_file.split('/')[-1]:
            valid_files.append(img_file)
    list.sort(valid_files, key=lambda x: int(x.split('/')[-1].split('_')[0]))

    with open('image_list.txt', 'w') as file:
        for item in valid_files:
            file.write("%s\n" % item)

    os.system('convert @image_list.txt {}.gif'.format(gif_name))

