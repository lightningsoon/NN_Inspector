import cv2
import os
import numpy as np
import torch
import torch.nn as nn


class Model_w_GradCAM():
    def __init__(self, model: torch.nn.Module, category_index: int = None, aimed_module: str = None):
        # 给了model，就知道了默认要取的layer，输出类别数。
        self.model = model
        self.model_items = []
        self.get_model_reversed_layers(model)
        self.model_items.reverse()
        self.get_classes()
        self.chose_module(aimed_module)
        self.set_class_index(category_index)
        self.set_hook()
        pass

    def get_model_reversed_layers(self, perspective_model):
        for name, module in perspective_model._modules.items():
            if len(module._modules) > 0:
                self.get_model_reversed_layers(module)
            else:
                self.model_items.append([name, module])
    def set_hook(self):
        def forward_hook(module, input, output):
            self.feature_map = output.detach().cpu()  # bs,channels,size,size

        def backward_hook(module, grad_in, grad_out):
            self.grad_map = grad_out[0].detach().cpu()

        self.aimed_module.register_forward_hook(forward_hook)
        self.aimed_module.register_backward_hook(backward_hook)

    def get_classes(self):
        # 数有多少类
        last_layer = self.model_items[0][1]
        self.num_classes = last_layer.out_features

    def chose_module(self, aimed_module):
        # 选择要可视化的最后一个卷积层，有值就按名字选
        module = None
        for name, module in self.model_items:
            if not aimed_module:
                if isinstance(module, (torch.nn.modules.conv._ConvNd,)):
                    break
            else:
                if name == aimed_module:
                    break
        assert module != None
        self.aimed_module = module

    def set_class_index(self, category_index):
        # 设置固定类别
        if not category_index:
            self.category_index = None
        else:
            assert isinstance(category_index, int)
            assert category_index < self.num_classes
            self.category_index = category_index

    def draw_cam(self, imgs, preds, category_index=None) -> list:
        # imgs: RGB
        # preds: shape=1,c
        # batch预测需要指定类别
        # 求梯度
        if not isinstance(imgs, (list,)) or isinstance(imgs, (np.ndarray,)):
            imgs = [imgs]
        elif isinstance(imgs, (list,)):
            pass
        else:
            raise TypeError
        self.model.zero_grad()
        if category_index:
            self.set_class_index(category_index)  # 后指定/重设置
        assert len(imgs) == preds.shape[0]
        if not self.category_index:
            # 没有类别，就按最大值来
            preds = preds[:, torch.argmax(preds, 1)]
        else:
            preds = preds[:, self.category_index]
        # 必须独立求梯度,只能传一张
        class_loss = torch.sum(preds)
        class_loss.backward(retain_graph=True)
        # 可视化图
        self.grad_map = torch.mean(self.grad_map, [2, 3], keepdim=True)  # mb,c,1,1
        cam = self.grad_map * self.feature_map  # mb,c,mH,mW
        cam = torch.sum(cam, 1).numpy()  # mb,mH,mW

        heatmaps = []
        for i in range(len(cam)):
            hm = self.heatmap(imgs[i], cam[i])
            heatmaps.append(hm)
        return heatmaps

    def heatmap(self, img, cam):
        img = np.float32(img) / 255
        cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
        # cam = np.maximum(cam, 0)# no elements is lower than zero.
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        # 附着
        heatmap = heatmap[..., ::-1] * 0.4 + np.float32(img)
        heatmap = heatmap / np.max(heatmap)
        heatmap = np.uint8(heatmap * 255)
        return heatmap

    def __call__(self, *args, **kwargs):
        preds = self.model(*args, **kwargs)  # softmax之前
        return preds





if __name__ == '__main__':
    from model import Net, img_preprocess

    print('for example!')

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    path_img = os.path.join(BASE_DIR, "cam_img", "test_img_8.png")
    path_net = os.path.join(BASE_DIR, "paras.pkl")
    output_dir = os.path.join(BASE_DIR, "result")

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    img = cv2.imread(path_img, 1)  # H*W*C
    img = cv2.resize(img, (32, 32))
    img = img[:, :, ::-1]  # BGR --> RGB

    # single mode
    img_input = img_preprocess(img)
    net = Net()
    net.load_state_dict(torch.load(path_net))
    net = Model_w_GradCAM(net)
    output = net(img_input)
    print(classes[torch.argmax(output.cpu(), 1)])
    cam = net.draw_cam([img], output)[0]
    from matplotlib import pyplot as plt

    plt.imshow(cam), plt.show()
    plt.imsave(os.path.join(output_dir, 'gradcam4ship.png'), cam)
    plt.imshow(img), plt.show()
    # plt.imsave(os.path.join(output_dir,'img.png'),img)

    # batch mode
    imgs = [img, img]
    img_input = img_preprocess(imgs)
    output = net(img_input)
    cam = net.draw_cam(imgs, output, 1)
    print(classes[1], 'number', len(cam))
    # plt.imshow(cam[0]), plt.show()
