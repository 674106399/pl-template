from config import cfg
from torchvision import transforms

tfms = {
    'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(90),
            transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
            transforms.Resize(cfg.input_img_shape),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            # transforms.Normalize()
            ]),
    'val':transforms.Compose([
            transforms.Resize(cfg.input_img_shape),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            # transforms.Normalize()
            ])
        }

def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

def Padding_resize(img):
    w, h = img.size
    pad_h = 0 if w < h else (w-h)//2
    pad_w = 0 if h < w else (h-w)//2

    tfm = transforms.Pad((pad_w,pad_h))

    return tfm(img)