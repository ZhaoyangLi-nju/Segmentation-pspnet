import os
import socket
from datetime import datetime


class RESNET18_SUNRGBD_CONFIG:

    def args(self):
        log_dir = '/home/lzy/summary/'
        # args = {'ROOT_DIR': '/home/lzy/summary'}
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')

        ########### Quick Setup ############
        model = 'PSP'

        task_name = 'resnet50_with_semanticBranch_'
        lr_schedule = 'lambda'  # lambda|step|plateau1
        pretrained = 'imagenet'
        content_pretrained = 'imagenet'
        gpus = '0,1,2,3,4,5,6,7'  # gpu no. you can add more gpus with comma, e.g., '0,1,2'
        batch_size = 8

        no_trans = False  # if True, no translation loss
        target_modal = 'seg'
        # target_modal = 'depth'
        if no_trans:
            loss = ['CLS']
        else:
            loss = ['CLS', 'SEMANTIC']



        unlabeld = False  # True for training with unlabeled data
        evaluate = True  # report mean acc after each epoch
        content_layers = '0,1,2,3,4'  # layer-wise semantic layers, you can change it to better adapt your task       
        alpha_content = 0.5

        multi_scale = False
        # multi_targets = ['depth']
        multi_targets = ['seg']
        multi_modal = False
        which_score = 'up'
        norm = 'in'

        len_gpu = str(len(gpus.split(',')))

        resume = False
        resume_path = 'FCN/2019_09_17_13_50_34/FCN_AtoB_5000.pth'

        log_path = os.path.join(log_dir, model, content_pretrained,
                                ''.join(
                                    [task_name, '_', 'alpha_', str(alpha_content), '_', which_score, '_', 'norm_', norm,
                                     '_', 'gpus-', len_gpu]), current_time)
        log_path = os.path.join(log_dir, model, content_pretrained,
                                ''.join(
                                    [task_name,'_', 'gpus-', len_gpu]), current_time)

        return {

            'MODEL': model,
            'GPU_IDS': gpus,
            'BATCH_SIZE': batch_size,
            'PRETRAINED': pretrained,

            'LOG_PATH': log_path,
            'data_dir': '/data0/lzy/SUNRGBD',

            # MODEL
            'ARCH': 'resnet18',
            'SAVE_BEST': True,
            'NO_TRANS': no_trans,
            'LOSS_TYPES': loss,

            #### DATA
            'NUM_CLASSES': 19,
            'UNLABELED': unlabeld,

            # TRAINING / TEST
            'RESUME': resume,
            'INIT_EPOCH': True,
            'RESUME_PATH': resume_path,
            'LR_POLICY': lr_schedule,

            'NITER': 5000,
            'NITER_DECAY': 60000,
            'NITER_TOTAL': 70000,
            'FIVE_CROP': False,
            'EVALUATE': evaluate,

            # translation task
            'WHICH_CONTENT_NET': 'resnet50v2',
            'CONTENT_LAYERS': content_layers,
            'CONTENT_PRETRAINED': content_pretrained,
            'ALPHA_CONTENT': alpha_content,
            'TARGET_MODAL': target_modal,
            'MULTI_SCALE': multi_scale,
            'MULTI_TARGETS': multi_targets,
            'WHICH_SCORE': which_score,
            'MULTI_MODAL': multi_modal,
            'UPSAMPLE_NORM': norm
        }
