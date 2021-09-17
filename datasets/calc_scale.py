import math

from torchvision.transforms.functional import scale

downfactor_dict = {
    'vgg19': {'kernel':[2,2,2,2],'stride':[2,2,2,2],'padding':[0,0,0,0]},
    'vgg19_bn': {'kernel':[2,2,2,2],'stride':[2,2,2,2],'padding':[0,0,0,0]},
    'vgg19_bag': {'kernel':[3,3,3,3,3],'stride':[1,2,2,2,2],'padding':[0,0,0,0,0]},
    'vgg19_bag_bn': {'kernel':[3,3,3,3,3],'stride':[1,2,2,2,2],'padding':[0,0,0,0,0]},
    'resnet50': {'kernel':[7,3,3,3,3],'stride':[2,2,2,2,2],'padding':[3,1,1,1,1]},
    'bagnet33': {'kernel':[3,3,3,3,3],'stride':[1,2,2,2,1],'padding':[0,0,0,0,0]},
    'bagnet17': {'kernel':[3,3,3,3,1],'stride':[1,2,2,2,1],'padding':[0,0,0,0,0]},
    'bagnet9': {'kernel':[3,3,3,1,1],'stride':[1,2,2,2,1],'padding':[0,0,0,0,0]},
}

def calc_layer_out_size(size, idx, arch):
    out_size = math.floor(((size - downfactor_dict[arch]['kernel'][idx] + 2*downfactor_dict[arch]['padding'][idx]) / downfactor_dict[arch]['stride'][idx]) + 1)
    return out_size

def calc_out_size(arch: str, in_size: tuple, pool_num: int, up_scale: int):
    """
    Return:
        out_size: tuple (w, h)
    """
    w, h = in_size

    if arch == 'fusionnet':
        bag_w = w
        res_w = w
        bag_h = h
        res_h = h
        for i in range(0, pool_num):
            if i < pool_num:
                bag_w = calc_layer_out_size(bag_w, i, 'bagnet33')
                res_w = calc_layer_out_size(res_w, i, 'resnet50')
                bag_h = calc_layer_out_size(bag_h, i, 'bagnet33')
                res_h = calc_layer_out_size(res_h, i, 'resnet50')                
                if i > 0:
                    w = int((bag_w + res_w) / 2)
                    h = int((bag_h + res_h) / 2)

    else:
        for i in range(0, pool_num):
            if i < pool_num:
                w = calc_layer_out_size(w, i, arch)
                h = calc_layer_out_size(h, i, arch)

    out_size = (w*up_scale, h*up_scale)
    return out_size

def calc_scale_factor(in_size: tuple, out_size: tuple):
    """
    Return:
        scale_factor: tuple (w_scale_factor, h_scale_factor)
    """
    in_w, in_h = in_size
    out_w, out_h = out_size

    scale_factor = (in_w / out_w, in_h / out_h)
    return scale_factor

if __name__ == '__main__':
    in_size = (512, 512)
    arch = 'vgg19_bag_bn'
    pool_num = 5
    up_scale = 1

    out_size = calc_out_size(arch, in_size, pool_num, up_scale)
    scale_factor = calc_scale_factor(in_size, out_size)

    print('output size: {}, scale factor: {}'.format(out_size, scale_factor))