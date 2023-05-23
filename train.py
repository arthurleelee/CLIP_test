import os
import argparse
import math
import csv
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import clip
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class image_title_dataset(Dataset):
    def __init__(self, list_image_path, list_txt, preprocess):
        self.image_path = list_image_path
        self.title  = clip.tokenize(list_txt) #you can tokenize everything at once in here(slow at the beginning), or tokenize it in the training loop.
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        image = self.preprocess(Image.open(self.image_path[idx])) # Image from PIL module
        title = self.title[idx]
        return image, title


def select_device(device):
    """Set devices' information to the program.
    Args:
        device: a string, like 'cpu' or '1,2,3,4'
    Returns:
        torch.device
    """
    if device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print('Using CPU for training... ')
    elif device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        assert torch.cuda.is_available()
        nd = len(device.strip().split(','))
        print(f'Using {nd} GPU for training... ')
    cuda = device != 'cpu' and torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    return device

def set_random_seed(seed, deterministic=False):
    """ Set random state to random libray, numpy, torch and cudnn.
    Args:
        seed: int value.
        deterministic: bool value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.deterministic = False
        cudnn.benchmark = True

#https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model): 
    for p in model.parameters():
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float()

def get_args_parser():
    # Arguments
    parser = argparse.ArgumentParser()
    # Model arguments
    parser.add_argument('--image_encoder', type=str, default="ViT-L/14")
    parser.add_argument('--adapter', action='store_true')

    # Datasets and loaders
    parser.add_argument('--kitti_image_file_path', type=str, default="../KITTI_DATASET_ROOT/training/image_2/")
    parser.add_argument('--label_2_sentence_file_path', type=str, default="./label_2_sentence.csv")
    parser.add_argument('--ratio', type=float, default=0.8)
    parser.add_argument('--batch_size', type=int, default=8)
    #parser.add_argument('--val_batch_size', type=int, default=8)

    # Optimizer and scheduler
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.2)
    parser.add_argument('--eps', type=float, default=1e-6)
    parser.add_argument('--beta_1', type=float, default=0.9)
    parser.add_argument('--beta_2', type=float, default=0.98)

    # Training
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--checkpoint_save_dir', type=str, default='./')
    parser.add_argument('--device', default='0', type=str, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--epoch', type=int, default=120)
    return parser.parse_args()

def main(args):
    device = select_device(args.device)
    set_random_seed(args.seed, deterministic=True)
    
    model, preprocess = clip.load(args.image_encoder, device=device, jit=False, adapter=args.adapter) #Must set jit=False for training
    
    if args.adapter:
        for name, param in model.named_parameters():
            name_split = name.split(".")
            if "adapter" not in name_split:
                param.requires_grad = False
            else:
                param.requires_grad = True
            if name == "ln_final.weight" or name == "ln_final.bias":
                param.requires_grad = True
            print("name: ", name)
            print("requires_grad: ", param.requires_grad)

    # use your own data
    image_file_list = [file_name for file_name in os.listdir(args.kitti_image_file_path)]
    image_file_list.sort()
    list_image_path = [args.kitti_image_file_path + i for i in image_file_list]
    
    with open(args.label_2_sentence_file_path, 'r', newline="") as sentence_file:
        rows = csv.reader(sentence_file)
        list_txt = [row[1] for row in rows]
    
    train_num = math.floor(len(list_image_path) * args.ratio)
    print("Train data: ", train_num)
    print("Val data: ", len(list_image_path) - train_num)
    dataset = image_title_dataset(list_image_path[:train_num], list_txt[:train_num], preprocess)
    train_dataloader = DataLoader(dataset, batch_size = args.batch_size) #Define your own dataloader

    if device == "cpu":
        model.float()
    else :
        clip.model.convert_weights(model) # Actually this line is unnecessary since clip by default already on float16

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(args.beta_1, args.beta_2), eps=args.eps, weight_decay=args.weight_decay) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

    # add your own code to track the training progress.
    model.train()
    for epoch in range(args.epoch):
        for batch in tqdm(train_dataloader) :
            optimizer.zero_grad()

            images, texts = batch 
        
            images = images.to(device)
            texts = texts.to(device)
        
            logits_per_image, logits_per_text = model(images, texts)

            ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
            
            total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth))/2
            total_loss.backward()
            
            if device == "cpu" or args.adapter:
                optimizer.step()
            else : 
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)
                
            print('[Train] Epoch %04d | Total Loss %.6f' % (epoch, total_loss.item()))

    """
    image = preprocess(Image.open(args.kitti_image_file_path + "/000001.png")).unsqueeze(0).to(device)
    text = clip.tokenize(["There are 1 car and 1 truck in the image.", "There is 1 car in the image.", "There is 1 truck in the image."]).to(device)
    model.eval()
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
      
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    print("Label probs:", probs)
    """
    
    if args.adapter:
        torch.save({'epoch': epoch, 
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': total_loss,
                    }, args.checkpoint_save_dir + "saved_adapter_model_epoch_" + str(args.epoch) + ".pt")
    else:
        torch.save({'epoch': epoch, 
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': total_loss,
                    }, args.checkpoint_save_dir + "saved_model_epoch_" + str(args.epoch) + ".pt")

if __name__ == '__main__':
    args = get_args_parser()
    main(args)