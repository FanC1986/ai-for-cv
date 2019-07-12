#! usr/bin/env python
import cv2

path=r'D:\groceries\AI\CV-CNN\Lesson2\img\dragon-mother.jpg'
img=cv2.imread(path)
# check the values of Gaussion kernel
kernel=cv2.getGaussianKernel(7,5)
print(kernel)

# gaussian blur
g1_img=cv2.GaussianBlur(img,(7,7),5)
g2_img=cv2.sepFilter2D(img,-1,kernel,kernel)



# show the pictures
cv2.imshow('dragon-mother',img)
cv2.imshow('blur1',g1_img)
cv2.imshow('blur2',g2_img)

key=cv2.waitKey()
if 27==key:
    cv2.destroyAllWindows()
