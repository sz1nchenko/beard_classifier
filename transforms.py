from torchvision.transforms import Compose, ToTensor, Normalize, Resize

from config import IMAGE_SIZE


def transform():
    return Compose([
        Resize((IMAGE_SIZE, IMAGE_SIZE)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
    ])