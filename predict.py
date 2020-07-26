from models import CNN
from torchvision import transforms
import torch
from skimage import io


class Predictor:
    def __init__(self, ckpt_path):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.ToTensor()
        ])
        self.checkpoint = torch.load(ckpt_path)
        self.model = CNN()
        self.model.load_state_dict(state_dict=self.checkpoint['state_dict'])
        self.use_cuda = False
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()
            self.use_cuda = True

    def run(self, path):
        img = self.transform(io.imread(path))
        img.unsqueeze_(0)
        if self.use_cuda:
            img = img.cuda()
        with torch.no_grad():
            pred = self.model(img)
        return pred.item()
