from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--test_mode', required=True,
                            help='should be local or submission. If val_test, we will use 3/5 for training, '
                                 '1/5 for validation and 1/5 for test; if val, we will use 4/5 for training and 1/5 for validation')
        parser.add_argument('--display_freq', type=int, default=2000, help='frequency of showing training results on screen')
        parser.add_argument('--display_ncols', type=int, default=3, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
        parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=200, help='frequency of showing training results on console')
        parser.add_argument('--save_latest_freq', type=int, default=40000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=500, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--pool_size', type=int, default=0, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--val_epoch_freq', type=int, default=10,
                            help='frequency of validation at the end of epochs')
        parser.add_argument('--num_val', type=int, default=400, help='how many test images to run during validation')
        parser.add_argument('--eval_val', action='store_true', help='use eval mode during validation.')

        parser.add_argument('--base_name', type=str, default=None, help='the name of base model to be loaded for fine tuning')
        parser.add_argument('--finetuning', action='store_true', help='continue training: fine tune the base model')
        parser.add_argument('--feature_extract', action='store_true', help='only fine tune the last layer (mask output) if set to be true')
        parser.add_argument('--loss_to_use', type=str, default='l2',
                            help='the loss function to be used for segmentation training')
        parser.add_argument('--n_fold', type=int, default=5, help='n_fold cross-validation')
        parser.add_argument('--test_index', type=int, default=None,
                            help='this argument has different effects when argument test_mode is set to different mode,'
                                 'val: we will not do test, so this argument does not have effect; '
                                 'test: the test fold is set to be this value, in default it is set to the last fold'
                                 'val_test: we will do both val and test, the test fold is set to this value')

        self.isTrain = True
        return parser
