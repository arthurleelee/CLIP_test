import torch
import clip
import numpy as np
import os
from PIL import Image
from torchsummary import summary
from train import *

sentence_mapping_id = {}
use_class = ["car", "van", "truck"]

sentence_list = ["A photo of " + str(i) + " " + use_class[0] + ", " + str(j) + " " + use_class[1] + " and " + str(k) + " " + use_class[2] + "." for i in range(11) for j in range(11) for k in range(11)]

"""
# old version
sentence_one_class_one_quantity_list = ["There is 1 " + i + " in the image." for i in use_class]
sentence_one_class_many_quantity_list = ["There are " + str(j) + " " + i + "s" + " in the image." for i in use_class for j in range(2, 11)]
sentence_two_class_one_quantity_one_quantity_list = ["There are 1 " + use_class[i] + " and 1 " + use_class[j] + " in the image." for i in range(len(use_class)) for j in range(i + 1, len(use_class))]
sentence_two_class_one_quantity_many_quantity_list = ["There are 1 " + use_class[i] + " and " + str(k) + " " + use_class[j] + "s in the image." for i in range(len(use_class)) for j in range(i + 1, len(use_class)) for k in range(2, 11)]
sentence_two_class_many_quantity_one_quantity_list = ["There are " + str(j) + " " + use_class[i] + "s and 1 " + use_class[k] + " in the image." for i in range(len(use_class)) for j in range(2, 11) for k in range(i + 1, len(use_class))]
sentence_two_class_many_quantity_many_quantity_list = ["There are " + str(j) + " " + use_class[i] + "s and " + str(l) + " " + use_class[k] + "s in the image." for i in range(len(use_class)) for j in range(2, 11) for k in range(i + 1, len(use_class)) for l in range(2, 11)]
sentence_three_class_one_quantity_one_quantity_one_quantity_list = ["There are 1 " + use_class[0] + ", 1 " + use_class[1] + " and 1 " + use_class[2] + " in the image."]
sentence_three_class_one_quantity_one_quantity_many_quantity_list = ["There are 1 " + use_class[0] + ", 1 " + use_class[1] + " and " + str(i) + " " + use_class[2] + "s in the image." for i in range(2, 11)]
sentence_three_class_one_quantity_many_quantity_one_quantity_list = ["There are 1 " + use_class[0] + ", " + str(i) + " " + use_class[1] + "s and 1 " + use_class[2] + " in the image." for i in range(2, 11)]
sentence_three_class_many_quantity_one_quantity_one_quantity_list = ["There are " + str(i) + " " + use_class[0] + "s, 1 " + use_class[1] + " and 1 " + use_class[2] + " in the image." for i in range(2, 11)]
sentence_three_class_many_quantity_many_quantity_one_quantity_list = ["There are " + str(i) + " " + use_class[0] + "s, " + str(j) + " " + use_class[1] + "s and 1 " + use_class[2] + " in the image." for i in range(2, 11) for j in range(2, 11)]
sentence_three_class_many_quantity_one_quantity_many_quantity_list = ["There are " + str(i) + " " + use_class[0] + "s, 1 " + use_class[1] + " and " + str(j) + " " + use_class[2] + "s in the image." for i in range(2, 11) for j in range(2, 11)]
sentence_three_class_one_quantity_many_quantity_many_quantity_list = ["There are 1 " + use_class[0] + ", " + str(i) + " " + use_class[1] + "s and " + str(j) + " " + use_class[2] + "s in the image." for i in range(2, 11) for j in range(2, 11)]
sentence_three_class_many_quantity_many_quantity_many_quantity_list = ["There are " + str(i) + " " + use_class[0] + "s, " + str(j) + " " + use_class[1] + "s and " + str(k) + " " + use_class[2] + "s in the image." for i in range(2, 11) for j in range(2, 11) for k in range(2, 11)]

sentence_list = sentence_one_class_one_quantity_list + \
                sentence_one_class_many_quantity_list + \
                sentence_two_class_one_quantity_one_quantity_list + \
                sentence_two_class_one_quantity_many_quantity_list + \
                sentence_two_class_many_quantity_one_quantity_list + \
                sentence_two_class_many_quantity_many_quantity_list + \
                sentence_three_class_one_quantity_one_quantity_one_quantity_list + \
                sentence_three_class_one_quantity_one_quantity_many_quantity_list + \
                sentence_three_class_one_quantity_many_quantity_one_quantity_list + \
                sentence_three_class_many_quantity_one_quantity_one_quantity_list + \
                sentence_three_class_many_quantity_many_quantity_one_quantity_list + \
                sentence_three_class_many_quantity_one_quantity_many_quantity_list + \
                sentence_three_class_one_quantity_many_quantity_many_quantity_list + \
                sentence_three_class_many_quantity_many_quantity_many_quantity_list
"""

def get_args_parser():
    # Arguments
    parser = argparse.ArgumentParser()
    # Model arguments
    parser.add_argument('--image_encoder', type=str, default="ViT-B/32")
    parser.add_argument('--adapter', action='store_true')
    parser.add_argument('--prompt', action='store_true')
    parser.add_argument('--vpt_version', type=int, default='2')
    # Datasets and loaders
    parser.add_argument('--kitti_image_file_path', type=str, default="../KITTI_DATASET_ROOT/training/image_2/")
    parser.add_argument('--label_2_sentence_file_path', type=str, default="./label_2_sentence.csv")
    parser.add_argument('--ratio', type=float, default=0.8)
    parser.add_argument('--val_batch_size', type=int, default=128)

    # Training
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--checkpoint', type=str, default='./saved_model_epoch_180.pt')
    parser.add_argument('--device', default='0', type=str, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    return parser.parse_args()

def main(args):
    device = select_device(args.device)
    prompt_config={'flag':False}
    if args.prompt:
        prompt_config={'flag':True,'num_token':5,'mode':'shallow', 'dropout':float(0),'prompt_dim':768}
    model, preprocess = clip.load(args.image_encoder, device=device, jit=False, adapter=args.adapter, prompt=prompt_config)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if args.adapter:
        for name, param in model.named_parameters():
            if "adapter" in name or "ln_final" in name or "ln_post" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
            print("name: ", name)
            print("requires_grad: ", param.requires_grad)
    elif args.prompt:
        if(args.vpt_version==1):
            for name, param in model.named_parameters():
                if(name.__contains__('visual.transformer')):
                    if(name.__contains__('prompt_embeddings')):
                        pass
                    else:
                        param.requires_grad = False
        elif(args.vpt_version==2):
            for name, param in model.named_parameters():
                if(name.__contains__('transformer')):
                    if(name.__contains__('prompt_embeddings')):
                        pass
                    else:
                        param.requires_grad = False
    # use your own data
    image_file_list = [file_name for file_name in os.listdir(args.kitti_image_file_path)]
    image_file_list.sort()
    list_image_path = [args.kitti_image_file_path + i for i in image_file_list]
    
    with open(args.label_2_sentence_file_path, 'r', newline="") as sentence_file:
        rows = csv.reader(sentence_file)
        list_txt = [row[1] for row in rows]
        
    with open(args.label_2_sentence_file_path, 'r', newline="") as sentence_file:
        rows = csv.reader(sentence_file)
        class_txt = [row[2] for row in rows]
        
    remove_index = []
    for i in range(len(list_txt)):
        if list_txt[i] == "None":
            remove_index.append(i)
    remove_index.reverse()

    for i in remove_index:
        list_image_path.pop(i)
        list_txt.pop(i)
        class_txt.pop(i)
    
    train_num = math.floor(len(list_txt) * args.ratio)
    print("Train data: ", train_num)
    print("Val data: ", len(list_txt) - train_num)
    dataset = image_title_dataset(list_image_path[train_num:], list_txt[train_num:], preprocess)
    valid_dataloader = DataLoader(dataset, batch_size = args.val_batch_size) #Define your own dataloader

    texts = clip.tokenize(sentence_list).to(device)

    model.float()
    for idx, batch in enumerate(valid_dataloader):
        images, _ = batch
        images = images.to(device)

        if idx == 0:
            summary(model, images, texts)
            break

    preds_list = np.array([])
    gt_list = np.array(class_txt[train_num:], dtype=int)

    model.eval()
    with torch.no_grad():
        for batch in tqdm(valid_dataloader):
            images, _ = batch

            images = images.to(device)

            logits_per_image, logits_per_text = model(images, texts)
            
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            preds = np.argmax(probs.round(3), axis=1)

            preds_list = np.append(preds_list, preds)

    acc = (preds_list == gt_list).sum() / preds_list.shape[0]
    print(f'Accuracy: {acc:.4f}')

if __name__ == '__main__':
    args = get_args_parser()
    main(args)