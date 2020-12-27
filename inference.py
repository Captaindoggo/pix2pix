import torch
from torchvision import transforms, utils, datasets

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow



def show_tensor(img, text=None):
  to_pil = transforms.ToPILImage()
  img = to_pil(img)
  if text:
    print(text)
  imshow(img)
  plt.show()

def show_tensors(imgs, texts=None):
  to_pil = transforms.ToPILImage()
  fig = plt.figure(figsize=(15, 15))
  n = len(imgs)
  for i in range(n):
    a = fig.add_subplot(1, n, i+1)
    img = to_pil(imgs[i])
    imgplot = plt.imshow(img)
    if texts:
      a.set_title(texts[i])
  plt.show()


def prediction(model, A, B, device):    # takes a batches of images A and B, in train mode model will not take input with batch_size == 1
  denorm = transforms.Normalize(mean=(-1.0, -1.0, -1.0), std=(2.0, 2.0, 2.0))  #standard mod_dataset used in this project uses normalization
  A = A.to(device)                                                          #therefore we use revrese normalization before printing images
  with torch.no_grad():
    pred = model.generator(A)
  A = A.cpu()
  pred = pred.cpu()
  B = B
  print(A.shape, pred.shape, B.shape)
  for i in range(len(A)):
    dn_A = denorm(A[i])
    dn_pred = denorm(pred[i])
    dn_B = denorm(B[i])
    show_tensors([dn_A, dn_pred, dn_B], ['input', 'prediction', 'target'])