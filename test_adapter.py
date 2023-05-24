import torch
import clip
import numpy as np
from PIL import Image


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


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device, jit=False, adapter=True)
checkpoint = torch.load("saved_adapter_model_epoch_120.pt")

model.load_state_dict(checkpoint['model_state_dict'])

image = preprocess(Image.open("../KITTI_DATASET_ROOT/training/image_2/006478.png")).unsqueeze(0).to(device)
text = clip.tokenize(sentence_list).to(device)

model.eval()
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs.round(3))
print("Label Prediction:", np.argmax(probs.round(3), axis=1))