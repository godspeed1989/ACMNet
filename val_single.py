import torch.nn
import numpy as np
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from options.test_options import TestOptions
from data import create_dataloader
from models import create_model
import util
from val import ToFalseColors

if __name__ == '__main__':
    opt = TestOptions().parse()
    data_loader = create_dataloader(opt)
    num_samples = len(data_loader)
    print('#test images = %d' % num_samples)

    model = create_model(opt)
    model.setup(opt)
    total_steps = 0
    model.eval()

    if opt.save:
        if opt.suffix != '':
            opt.suffix = '_' + opt.suffix
        dirs = os.path.join('results', opt.model+opt.suffix)
        os.makedirs(dirs)

    print('load model done')

    data = {}
    s_depth = cv2.imread('./0000000001sparse.png', cv2.IMREAD_UNCHANGED)
    img = cv2.imread('./0000000001img.png', cv2.IMREAD_COLOR)

    with open('./0000000001.txt', 'r') as f:
        calib = f.readline()
        calib = calib.splitlines()[0].rstrip().split(' ')
    K = np.zeros((3, 3), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            K[i, j] = float(calib[i*3+j])

    # bottom crop
    h, w = img.shape[0], img.shape[1]
    H = 352
    s = int(round(w - 1216) / 2)
    img = img[h-H:, s:s+1216]
    s_depth = s_depth[h-H:, s:s+1216]
    K[0, 2] = K[0, 2] - s
    K[1, 2] = K[1, 2] - (h-H)
    # to tensor
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0]))
    img = img.astype(np.float32)
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    data['img'] = torch.tensor(img).unsqueeze(0).to(device)
    s_depth = np.expand_dims(s_depth, axis=0)
    s_depth = s_depth.astype(np.float32) / 256.0 # (0..85)
    data['sparse'] = torch.tensor(s_depth).unsqueeze(0).to(device)
    data['K'] = torch.tensor(K).unsqueeze(0).to(device)

    model.set_input(data)
    model.test()

    visuals = model.get_current_visuals()
    pred_depth = np.squeeze(visuals['pred'].data.cpu().numpy())
    s_depth = np.squeeze(data['sparse'].data.cpu().numpy())

    fig = plt.figure()
    fig.add_subplot(2, 1, 1)
    plt.imshow(pred_depth)
    fig.add_subplot(2, 1, 2)
    plt.imshow(s_depth)
    plt.show()

    pred_image = ToFalseColors(pred_depth)
    s_image = ToFalseColors(s_depth, mask=(s_depth>0).astype(np.float32))

    dirs = os.path.join('results', opt.model+opt.suffix)
    os.makedirs(dirs, exist_ok=True)
    ind = 1
    pred_img = Image.fromarray(pred_image, 'RGB')
    s_img = Image.fromarray(s_image, 'RGB')
    pred_img.save('%s/%05d_pred.png'%(dirs, ind))
    s_img.save('%s/%05d_sparse.png'%(dirs, ind))
    im = util.tensor2im(visuals['img'])
    b, g ,r =cv2.split(im)
    im = cv2.merge([r,g,b])
    util.save_image(im, '%s/%05d_img.png'%(dirs, ind), 'RGB')
