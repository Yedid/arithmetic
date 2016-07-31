from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os, sys

def get_img(n, img_sz):
  im = np.zeros(img_sz)
  pil_im = Image.fromarray(im)
  draw = ImageDraw.Draw(pil_im)
  font = ImageFont.truetype("DejaVuSans.ttf", 12)
  sz = font.getsize(str(n))
  draw.text((img_sz[1]-sz[0], 2), str(n), font=font)
  im = np.asarray(pil_im)
  return im

def get_action(n1, n2):
  return n1 + n2

def get_dataset(NData, numberfile, NMax, img_sz):    
  data = np.zeros((NData,img_sz[0],img_sz[1],2))
  label = np.zeros((NData,img_sz[0]*img_sz[1]))
  f = open(numberfile, 'w')
  for i in range(NData):
    n1 = np.random.randint(0,NMax)
    n2 = np.random.randint(0,NMax)
    nres = get_action(n1,n2)
    data[i,:,:,0] = get_img(n1, img_sz)
    data[i,:,:,1] = get_img(n2, img_sz)
    label[i,:] = get_img(nres, img_sz).flatten()
    f.write(str(n1) + " " + str(n2) + " " + str(nres) + "\n")  
    if i%100 == 0:
      print "Frame: " + str(i)
  f.close()
  return data, label

def main():
  NMax = 4999999
  img_sz = (15,60)

  print "Generate data:"
  train_data, train_label = get_dataset(30000, 'train_numbers.txt', NMax, img_sz)
  test_data, test_label = get_dataset(30000, 'test_numbers.txt', NMax, img_sz)

  mu = np.mean(train_data,axis=0)[None,:,:,:]
  np.save("mu", mu)
  train_data -= mu
  test_data -= mu

  rp = np.random.permutation(train_data.shape[0])
  train_data = train_data[rp]
  train_label = train_label[rp]

  np.save('train_X',train_data)
  np.save('train_y',train_label)
  np.save('test_X',test_data)
  np.save('test_y',test_label)

if __name__ == "__main__":
  main()