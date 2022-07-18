import os
import csv
import torch
import torch.nn as nn
import clip
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
model, preprocess = clip.load("ViT-L/14",device=device,jit=False) #Must set jit=False for training
BATCH_SIZE = 8
EPOCH = 8

class image_title_dataset(Dataset):
    def __init__(self, list_image_path, list_txt):

        self.image_path = list_image_path
        self.title  = clip.tokenize(list_txt) #you can tokenize everything at once in here(slow at the beginning), or tokenize it in the training loop.

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        image = preprocess(Image.open(self.image_path[idx])) # Image from PIL module
        title = self.title[idx]
        return image, title

# use your own data
image_file_list = [file_name for file_name in os.listdir("/home/arthur/Downloads/KITTI_DATASET_ROOT/training/image_2/")]
image_file_list.sort()
list_image_path = ["/home/arthur/Downloads/KITTI_DATASET_ROOT/training/image_2/" + i for i in image_file_list]
with open("/home/arthur/Downloads/KITTI_DATASET_ROOT/training/label_2_sentence.csv", 'r', newline="") as sentence_file:
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
dataset = image_title_dataset(list_image_path, list_txt)
train_dataloader = DataLoader(dataset, batch_size = BATCH_SIZE) #Define your own dataloader

#https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 


if device == "cpu":
  model.float()
else :
  clip.model.convert_weights(model) # Actually this line is unnecessary since clip by default already on float16

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

# add your own code to track the training progress.
model.train()
for epoch in range(EPOCH):
  for batch in tqdm(train_dataloader) :
      optimizer.zero_grad()

      images,texts = batch 
    
      images= images.to(device)
      texts = texts.to(device)
    
      logits_per_image, logits_per_text = model(images, texts)

      ground_truth = torch.arange(len(images),dtype=torch.long,device=device)

      total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
      total_loss.backward()
      if device == "cpu":
         optimizer.step()
      else : 
        convert_models_to_fp32(model)
        optimizer.step()
        clip.model.convert_weights(model)

image = preprocess(Image.open("/home/arthur/Downloads/KITTI_DATASET_ROOT/training/image_2/000001.png")).unsqueeze(0).to(device)
text = clip.tokenize(["There are 1 car and 1 truck in the image.", "There is 1 car in the image.", "There is 1 truck in the image."]).to(device)
model.eval()
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)

model_save_dir = "./saved_model"
torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss,
        }, model_save_dir + "_epoch_" + str(EPOCH) + ".pt")