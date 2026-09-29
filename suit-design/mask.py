from PIL import Image; import numpy as np
from scipy import ndimage as nd
a=np.asarray(Image.open('template.jpg').convert('RGB')).astype(int)
g=a.mean(2)
m=g<249
m=nd.binary_closing(m,iterations=3)
m=nd.binary_fill_holes(m)
m=nd.binary_opening(m,iterations=2)
lab,n=nd.label(m); sizes=nd.sum(m,lab,range(1,n+1))
keep=np.isin(lab,1+np.where(sizes>5000)[0])
np.save('mask.npy',keep)
Image.fromarray((keep*255).astype(np.uint8)).save('mask.png')
print(n, sorted(sizes)[-5:])
