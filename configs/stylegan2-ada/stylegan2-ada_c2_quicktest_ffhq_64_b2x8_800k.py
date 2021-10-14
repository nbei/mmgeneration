"""Config for the `config-f` setting in StyleGAN2."""

_base_ = [
    '../_base_/datasets/ffhq_flip.py', '../_base_/models/stylegan2_base.py',
    '../_base_/default_runtime.py'
]

aug_kwargs = {
    'xflip': 1,
    'rotate90': 1,
    'xint': 1,
    'scale': 1,
    'rotate': 1,
    'aniso': 1,
    'xfrac': 1,
    'brightness': 1,
    'contrast': 1,
    'lumaflip': 1,
    'hue': 1,
    'saturation': 1
}

d_reg_interval = 16.
model = dict(
    generator=dict(out_size=64),
    discriminator=dict(
        type='ADAStyleGAN2Disc',
        in_size=64,
        data_aug=dict(type='ADAAug', aug_pipeline=aug_kwargs),
    ),
    disc_auxiliary_loss=[
        dict(
            type='R1GradientPenalty',
            loss_weight=10. / 2. * d_reg_interval,
            interval=d_reg_interval,
            norm_mode='HWC',
            data_info=dict(real_data='real_imgs', discriminator='disc')),
        dict(
            type='AdaUpdater',
            data_info=dict(
                disc='disc',
                real_logit='disc_pred_real',
                iteration='iteration'))
    ],
)

dataset_type = 'QuickTestImageDataset'
data = dict(
    samples_per_gpu=2,
    train=dict(type=dataset_type, size=(64, 64)),
    val=dict(type=dataset_type, size=(64, 64)))

ema_half_life = 10.  # G_smoothing_kimg

custom_hooks = [
    dict(
        type='VisualizeUnconditionalSamples',
        output_dir='training_samples',
        interval=5000),
    dict(
        type='ExponentialMovingAverageHook',
        module_keys=('generator_ema', ),
        interval=1,
        interp_cfg=dict(momentum=0.5**(32. / (ema_half_life * 1000.))),
        priority='VERY_HIGH')
]

metrics = dict(
    fid50k=dict(
        inception_pkl='work_dirs/inception_pkl/ffhq-64_50k_rgb.pkl',
        bgr2rgb=True))

checkpoint_config = dict(interval=10000, by_epoch=False, max_keep_ckpts=30)
lr_config = None

log_config = dict(interval=2, hooks=[dict(type='TextLoggerHook')])

total_iters = 16
