from crop import cropper
import cv2
import torch

class cropCUB:

    def __init__(self, mode='train'):
        f = open('./CUB_200_2011/train_test_split.txt', 'r')
        l = open('./CUB_200_2011/image_class_labels.txt', 'r')
        self.mode = mode
        data = []
        label = []
        while True:
            lines = f.readline()
            linel = l.readline()
            if not lines:
                break
            lines = lines.split()
            linel = linel.split()
            if mode == 'train':
                m = '1'
            if mode == 'test':
                m = '0'
            if lines[1] == m:
                data.append(int(linel[0]) - 1)
                label.append(int(linel[1]) - 1)
        self.inputs = data
        self.label = label
        self.dataCropper = cropper()

    def __getitem__(self, index):
        aug = index >= len(self.inputs)
        index = index % len(self.inputs)
        image_id = self.inputs[index]
        d = []
        for p in range(15):
            crop = self.dataCropper.crop(image_id, p)
            if aug:
                crop = cv2.flip(crop, 1)
            d.append(crop)
        return torch.Tensor(d), self.label[index]
    
    def __len__(self):
        if self.mode == 'train':
            return len(self.inputs) * 2
        else:
            return len(self.inputs)
        
