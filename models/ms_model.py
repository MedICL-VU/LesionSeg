import torch
import torch.nn as nn
from util.image_pool import ImagePool
from .base_model import BaseModel
from .losses import FocalLoss, dice_loss
from . import networks
from configurations import *


class MsModel(BaseModel):
    def name(self):
        return 'MsModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        if is_train:
            parser.add_argument('--lambda_L2', type=int, default=2000, help='weight for L2 loss')
            parser.add_argument('--lambda_dice', type=int, default=100, help='weight for dice loss')
            parser.add_argument('--lambda_focal', type=int, default=10000, help='weight for focal loss')

        return parser

    def initialize(self, opt, model_suffix):
        BaseModel.initialize(self, opt, model_suffix)
        self.isTrain = opt.isTrain

        self.model_names = ['G']
        self.netG = networks.define_G(opt.input_nc * len(MODALITIES), opt.init_type, opt.init_gain, self.gpu_ids)
        self.visual_names = ['real_mask', 'fake_mask']
        for modality in MODALITIES:
            self.visual_names += [modality]

        if self.isTrain:
            self.loss_names = ['total']
            self.criterion_names = []
            criterions = {'L2': torch.nn.MSELoss(), 'focal': FocalLoss(gamma=1, alpha=0.25).to(self.device), 'dice': dice_loss}
            for k in criterions.keys():
                if k in opt.loss_to_use:
                    self.loss_names += [k]
                    setattr(self, 'criterion_%s' % k, criterions[k])
                    self.criterion_names.append(k)
            assert len(self.criterion_names), 'should use at least one loss function in L2, focal, dice'

            self.fake_AB_pool = ImagePool(opt.pool_size)

            # for feature extraction, update only the last layer, otherwise update all the parameters
            if self.opt.feature_extract:
                params_to_update = []
                print("Params to learn:")
                for name, param in self.netG.named_parameters():
                    if 'thres' in name:
                        params_to_update.append(param)
                    else:
                        param.requires_grad = False
            else:
                params_to_update = self.netG.parameters()
            self.optimizers = [torch.optim.Adam(params_to_update, lr=opt.lr, betas=(opt.beta1, 0.999))]

    def set_input(self, input):
        for modality in MODALITIES:
            setattr(self, modality, input[modality].to(self.device))
        self.real_mask = input['mask'].to(self.device)

    def forward(self):
        self.fake_mask = self.netG(torch.cat([getattr(self, k) for k in MODALITIES], 1))

    def backward_G(self):
        self.loss_total = 0
        fake_mask = (self.fake_mask + 1) / 2
        real_mask = (self.real_mask + 1) / 2
        for k, criterion_name in enumerate(self.criterion_names):
            criterion = getattr(self, 'criterion_%s' % criterion_name)
            if criterion_name == 'dice':
                tmp = criterion(fake_mask, real_mask) * self.opt.lambda_dice
            elif self.criterion_names[k] == 'focal':
                tmp = criterion(self.fake_mask, real_mask) * self.opt.lambda_focal
            else:
                tmp = criterion(self.fake_mask, self.real_mask) * self.opt.lambda_L2
            self.loss_total += tmp
            setattr(self, 'loss_%s' % self.criterion_names[k], tmp)

        self.loss_total.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizers[0].zero_grad()
        self.backward_G()
        self.optimizers[0].step()
