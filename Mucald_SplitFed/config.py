# config.py

class Config:
    def __init__(self):
        # -----------------------------------------------------------------------------
        # Data settings
        # -----------------------------------------------------------------------------
        self.DATA_BATCH_SIZE = 128
        self.DATA_PATH = ''
        self.DATA_DATASET = 'imagenet'
        self.DATA_IMG_SIZE = 224
        self.DATA_INTERPOLATION = 'bicubic'
        self.DATA_ZIP_MODE = False
        self.DATA_CACHE_MODE = 'part'
        self.DATA_PIN_MEMORY = True
        self.DATA_NUM_WORKERS = 8

        # -----------------------------------------------------------------------------
        # Model settings
        # -----------------------------------------------------------------------------
        self.MODEL_TYPE = 'swin'
        self.MODEL_NAME = 'swin_tiny_patch4_window7_224'
        self.MODEL_PRETRAIN_CKPT = './pretrained_ckpt/swin_tiny_patch4_window7_224.pth'
        self.MODEL_RESUME = ''
        self.MODEL_NUM_CLASSES = 1000
        self.MODEL_DROP_RATE = 0.0
        self.MODEL_DROP_PATH_RATE = 0.1
        self.MODEL_LABEL_SMOOTHING = 0.1

        # Swin Transformer parameters
        self.MODEL_SWIN_PATCH_SIZE = 4
        self.MODEL_SWIN_IN_CHANS = 3
        self.MODEL_SWIN_EMBED_DIM = 96
        self.MODEL_SWIN_DEPTHS = [2, 2, 6, 2]
        self.MODEL_SWIN_DECODER_DEPTHS = [2, 2, 6, 2]
        self.MODEL_SWIN_NUM_HEADS = [3, 6, 12, 24]
        self.MODEL_SWIN_WINDOW_SIZE = 7
        self.MODEL_SWIN_MLP_RATIO = 4.0
        self.MODEL_SWIN_QKV_BIAS = True
        self.MODEL_SWIN_QK_SCALE = None
        self.MODEL_SWIN_APE = False
        self.MODEL_SWIN_PATCH_NORM = True
        self.MODEL_SWIN_FINAL_UPSAMPLE = "expand_first"

        # -----------------------------------------------------------------------------
        # Training settings
        # -----------------------------------------------------------------------------
        self.TRAIN_START_EPOCH = 0
        self.TRAIN_EPOCHS = 300
        self.TRAIN_WARMUP_EPOCHS = 20
        self.TRAIN_WEIGHT_DECAY = 0.05
        self.TRAIN_BASE_LR = 5e-4
        self.TRAIN_WARMUP_LR = 5e-7
        self.TRAIN_MIN_LR = 5e-6
        self.TRAIN_CLIP_GRAD = 5.0
        self.TRAIN_AUTO_RESUME = True
        self.TRAIN_ACCUMULATION_STEPS = 0
        self.TRAIN_USE_CHECKPOINT = False

        self.TRAIN_LR_SCHEDULER_NAME = 'cosine'
        self.TRAIN_LR_SCHEDULER_DECAY_EPOCHS = 30
        self.TRAIN_LR_SCHEDULER_DECAY_RATE = 0.1

        self.TRAIN_OPTIMIZER_NAME = 'adamw'
        self.TRAIN_OPTIMIZER_EPS = 1e-8
        self.TRAIN_OPTIMIZER_BETAS = (0.9, 0.999)
        self.TRAIN_OPTIMIZER_MOMENTUM = 0.9

        # -----------------------------------------------------------------------------
        # Augmentation settings
        # -----------------------------------------------------------------------------
        self.AUG_COLOR_JITTER = 0.4
        self.AUG_AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
        self.AUG_REPROB = 0.25
        self.AUG_REMODE = 'pixel'
        self.AUG_RECOUNT = 1
        self.AUG_MIXUP = 0.8
        self.AUG_CUTMIX = 1.0
        self.AUG_CUTMIX_MINMAX = None
        self.AUG_MIXUP_PROB = 1.0
        self.AUG_MIXUP_SWITCH_PROB = 0.5
        self.AUG_MIXUP_MODE = 'batch'

        # -----------------------------------------------------------------------------
        # Testing settings
        # -----------------------------------------------------------------------------
        self.TEST_CROP = True

        # -----------------------------------------------------------------------------
        # Misc
        # -----------------------------------------------------------------------------
        self.AMP_OPT_LEVEL = ''
        self.OUTPUT = ''
        self.TAG = 'default'
        self.SAVE_FREQ = 1
        self.PRINT_FREQ = 10
        self.SEED = 0
        self.EVAL_MODE = False
        self.THROUGHPUT_MODE = False
        self.LOCAL_RANK = 0


# Usage:
# from config import Config
# config = Config()
