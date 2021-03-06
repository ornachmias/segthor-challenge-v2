def get_full_model(model_name='deeplabv3_plus_resnet', loss_name='DiceLoss2', n_classes=5, alpha=None,
                   if_closs=1, pretrained=True, class_weights=None):

    if loss_name == 'CombinedLoss':
        from .loss_funs import CombinedLoss
        loss = CombinedLoss(alpha=alpha, if_closs=if_closs)
    elif loss_name == 'DicePlusXEnt':
        from .loss_funs import DicePlusXEntLoss
        loss = DicePlusXEntLoss(class_weights=class_weights)
    elif loss_name == 'WeightedCombinedLoss':
        from .loss_funs import WeightedCombinedLoss
        loss = WeightedCombinedLoss(weights=class_weights, alpha=alpha, if_closs=if_closs)

    if model_name == 'ResUNet101':
        from .my_unet import ResUNet101
        net = ResUNet101(n_channels=3, n_classes=n_classes, drop_rate=0., pretrained=pretrained)
    elif model_name == 'ResUNet101Index':
        from .my_unet import ResUNet101Index
        net = ResUNet101Index(n_channels=4, n_classes=n_classes, drop_rate=0., pretrained=pretrained)
    elif model_name == 'ResUNet101NoBatchNorm':
        from models.resnet_no_norm import ResUNet101NoBatchNorm
        net = ResUNet101NoBatchNorm(n_channels=3, n_classes=n_classes, drop_rate=0., pretrained=pretrained)
    elif model_name == 'ResUNet101Attention':
        from .my_unet import ResUNet101Attention
        net = ResUNet101Attention(n_channels=3, n_classes=n_classes, drop_rate=0., pretrained=pretrained)
    elif model_name == 'ResUNet152':
        from .my_unet import ResUNet152
        net = ResUNet152(n_channels=3, n_classes=n_classes, drop_rate=0., pretrained=pretrained)
    elif model_name == 'DenseUNet121':
        from .my_unet import DenseUNet121
        net = DenseUNet121(n_channels=3, n_classes=n_classes, drop_rate=0., pretrained=pretrained)    
    elif model_name == 'DenseUNet161':
        from .my_unet import DenseUNet161
        net = DenseUNet161(n_channels=3, n_classes=n_classes, drop_rate=0., pretrained=pretrained)
    elif model_name == 'DenseUNet201':
        from .my_unet import DenseUNet201
        net = DenseUNet201(n_channels=3, n_classes=n_classes, drop_rate=0., pretrained=pretrained)
    elif model_name == 'DenseUNet169':
        from .my_unet import DenseUNet169
        net = DenseUNet169(n_channels=3, n_classes=n_classes, drop_rate=0., pretrained=pretrained)
    return net, loss
