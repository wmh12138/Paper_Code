import argparse

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import vgg19
import numpy as np
from PAN_gradcam import GradCam

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Grad-CAM')
    parser.add_argument('--image_name', default='30.png', type=str, help='the tested image name')
    parser.add_argument('--save_name', default='30_HD_grad_cam.png', type=str, help='saved image name')
    parser.add_argument('--dropoutrate', type=float, default=0.3, help='dropoutrate')
    opt = parser.parse_args()

    IMAGE_NAME = opt.image_name
    SAVE_NAME = opt.save_name
    test_image = (transforms.ToTensor()(Image.open(IMAGE_NAME))).unsqueeze(dim=0)
    print(test_image.shape)
    img_shape = (1,28, 28)
    #加载模型类
    class Recognizer_CNN(nn.Module):
        def __init__(self):
            super(Recognizer_CNN, self).__init__()
            # dense1_bn = nn.BatchNorm1d(512)
            # dense2_bn = nn.BatchNorm1d(256)
            # 创建一个15层的卷积神经网络
            # 第一层是输出频道数为3，输入频道数为96，卷积核是3*3，其他为默认值
            # 输入32*32的RGB格式的图像，输出的是这个尺寸:torch.Size([1, 10, 14, 14])
            self.conv = nn.Sequential(
                nn.Conv2d(3, 84, 3),
                nn.BatchNorm2d(84),
                nn.ReLU(),
                # nn.BatchNorm2d(84),
                nn.Dropout(opt.dropoutrate),
                # nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(84, 84, 3, stride=2),
                nn.BatchNorm2d(84),
                nn.ReLU(),
                # nn.BatchNorm2d(84),
                nn.Dropout(opt.dropoutrate),
                # nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(84, 168, 1),
                nn.BatchNorm2d(168),
                nn.ReLU(),
                # nn.BatchNorm2d(168),
                nn.Dropout(opt.dropoutrate),
                # nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(168, 8, 1),
                nn.ReLU(),
                # nn.BatchNorm2d(8),
                nn.Dropout(opt.dropoutrate),
                # nn.LeakyReLU(0.2, inplace=True),
            )

            self.fc1 = nn.Sequential(
                nn.Linear(144 * 8, 1000),
                nn.LeakyReLU(0.2, inplace=True),
                # nn.Dropout(opt.dropoutrate),
                nn.Linear(1000, 1),
                nn.Sigmoid(),
            )

        def forward(self, img):
            conv_d = self.conv(img)
            out = self.fc1(conv_d.view(conv_d.shape[0], -1))
            return out

    #实例化一个模型并加载参数
    model = Recognizer_CNN()
    #PATH = 'D:\Desktop\GradCAM-master\HD_PAN_Blood_Model_params.pkl'
    #model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
    model = torch.load('D:\Desktop\GradCAM-master\HD_PAN_Blood_Model_18.pth',map_location=torch.device('cpu'))
    model.eval()

    if torch.cuda.is_available():
        test_image = test_image.cuda()
        model.cuda()
    grad_cam = GradCam(model)
    print(test_image.shape)
    feature_image = grad_cam(test_image).squeeze(dim=0)
    feature_image = transforms.ToPILImage()(feature_image)
    feature_image.save(SAVE_NAME)
