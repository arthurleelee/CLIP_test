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
        if p.requires_grad:
            p.data = p.data.float() 
            p.grad.data = p.grad.data.float()

def get_args_parser():
    # Arguments
    parser = argparse.ArgumentParser()
    # Model arguments
    parser.add_argument('--image_encoder', type=str, default="ViT-B/32")
    parser.add_argument('--adapter', action='store_true')

    # Datasets and loaders
    parser.add_argument('--kitti_image_file_path', type=str, default="../KITTI_DATASET_ROOT/training/image_2/")
    parser.add_argument('--label_2_sentence_file_path', type=str, default="./label_2_sentence.csv")
    parser.add_argument('--ratio', type=float, default=0.8)
    parser.add_argument('--batch_size', type=int, default=128)
    #parser.add_argument('--val_batch_size', type=int, default=8)

    # Optimizer and scheduler
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--eps', type=float, default=1e-6)
    parser.add_argument('--beta_1', type=float, default=0.9)
    parser.add_argument('--beta_2', type=float, default=0.98)

    # Training
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--checkpoint_save_dir', type=str, default='./')
    parser.add_argument('--device', default='0', type=str, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--epoch', type=int, default=180)
    return parser.parse_args()

def main(args):
    device = select_device(args.device)
    set_random_seed(args.seed, deterministic=True)
    
    model, preprocess = clip.load(args.image_encoder, device=device, jit=False, adapter=args.adapter) #Must set jit=False for training
    
    if args.adapter:
        for name, param in model.named_parameters():
            if "adapter" in name or "ln_final" in name or "ln_post" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
            print("name: ", name)
            print("requires_grad: ", param.requires_grad)

    # use your own data
    image_file_list = [file_name for file_name in os.listdir(args.kitti_image_file_path)]
    image_file_list.sort()
    list_image_path = [args.kitti_image_file_path + i for i in image_file_list]
    
    with open(args.label_2_sentence_file_path, 'r', newline="") as sentence_file:
        rows = csv.reader(sentence_file)
        list_txt = [row[1] for row in rows]
    remove_index = []
    for i in range(len(list_txt)):
        if list_txt[i] == "None":
            remove_index.append(i)
    remove_index.reverse()
    for i in remove_index:
        list_image_path.pop(i)
        list_txt.pop(i)
    
    train_num = math.floor(len(list_txt) * args.ratio)
    print("Train data: ", train_num)
    print("Val data: ", len(list_txt) - train_num)
    dataset = image_title_dataset(list_image_path[:train_num], list_txt[:train_num], preprocess)
    train_dataloader = DataLoader(dataset, batch_size = args.batch_size) #Define your own dataloader

    model.float()
    """
    if device == "cpu":
        model.float()
    else :
        clip.model.convert_weights(model) # Actually this line is unnecessary since clip by default already on float16
    """
    
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
   
    if args.adapter:
        other_learnable_parameters_list = list(map(id, model.visual.ln_post.parameters())) + list(map(id, model.ln_final.parameters()))
        adapter_parameters = filter(lambda p: p.requires_grad and id(p) not in other_learnable_parameters_list, model.parameters())
        other_learnable_parameters = filter(lambda p: p.requires_grad and id(p) in other_learnable_parameters_list, model.parameters())
        optimizer = optim.Adam([{'params':adapter_parameters, 'lr':5e-4, 'weight_decay':args.weight_decay}, 
                                {'params':other_learnable_parameters, 'lr':args.lr, 'betas':tuple([args.beta_1, args.beta_2]), 'eps':args.eps, 'weight_decay':args.weight_decay}])
        #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta_1, args.beta_2), eps=args.eps, weight_decay=args.weight_decay)
        #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

    # add your own code to track the training progress.
    for epoch in range(args.epoch):

        each_epoch_total_loss = 0
        each_epoch_image_acc = 0
        each_epoch_text_acc = 0

        for batch in tqdm(train_dataloader) :
            optimizer.zero_grad()

            images, texts = batch 
        
            images = images.to(device)
            texts = texts.to(device)
        
            logits_per_image, logits_per_text = model(images, texts)

            ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
            
            total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth))/2
            total_loss.backward()

            image_acc = (logits_per_image.argmax(dim=-1) == ground_truth).float()
            text_acc = (logits_per_text.argmax(dim=-1) == ground_truth).float()
            
            optimizer.step()
            """
            if device == "cpu":
                optimizer.step()
            else : 
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)
            """

            #print('[Train] Epoch %04d | Loss %.6f | Image Acc %.6f | Text Acc %.6f' % (epoch, total_loss.item(), image_acc.sum().item(), text_acc.sum().item()))
            each_epoch_total_loss = each_epoch_total_loss + total_loss.item()
            each_epoch_image_acc = each_epoch_image_acc + image_acc.sum().item()
            each_epoch_text_acc = each_epoch_text_acc + text_acc.sum().item()

        print('[Train] Epoch %04d | Total Loss %.6f | Image Acc %.6f | Text Acc %.6f' % (epoch, each_epoch_total_loss / len(train_dataloader), each_epoch_image_acc / len(train_dataloader), each_epoch_text_acc / len(train_dataloader)))

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