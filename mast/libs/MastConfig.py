import yaml


class MastConfig(object):
    def __init__(self, config_yml_path):
        file = open(config_yml_path)
        config = yaml.safe_load(file)
        self.use_seg = config['use_seg']
        self.resize = config['resize']
        self.type = config['type']
        self.encoder_path = config['encoder_path']
        self.decoder_r11_path = config['decoder_r11_path']
        self.decoder_r21_path = config['decoder_r21_path']
        self.decoder_r31_path = config['decoder_r31_path']
        self.decoder_r41_path = config['decoder_r41_path']
        self.decoder_r51_path = config['decoder_r51_path']
        self.layers = config['layers']
        self.orth_constraint = config['orth_constraint']
        self.post_smoothing = config['post_smoothing']
        self.fast = config['fast']
        self.gpu = config['gpu']
        self.device = config['device']
        self.max_use_num = config['max_use_num']
        self.soft_lambda = config['soft_lambda']
        self.k_cross = config['k_cross']
        self.patch_size = config['patch_size']
        self.style_weight = config['style_weight']
        self.reduce_dim_type = config['reduce_dim_type']
        self.dist_type = config['dist_type']
        self.dim_thresh = config['dim_thresh']
