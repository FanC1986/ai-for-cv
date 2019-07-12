#! usr/bin/env python3
import cv2
import random
import numpy as np
import matplotlib

'''
author : FanC86
date   : 2019-7-8
'''


class imagePpcess:

    def __init__(self,filename):
        self.filename=filename
        self.image=cv2.imread(filename)

    def crop_pic(self):
        self.image=self.image[0:700,:,:]

    def split_pic(self):
        self.imageR ,self.imageG,self.imageB= cv2.split(self.image)

    def rotate_pic(self):
        M = cv2.getRotationMatrix2D((self.image.shape[1] / 2, self.image.shape[0] / 2), 30, 1)  # center, angle, scale
        self.image_rotate = cv2.warpAffine(self.image, M, (self.image.shape[1], self.image.shape[0]))
        print(M)

    def affine_transform(self):
        rows, cols, ch = self.image.shape
        pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
        pts2 = np.float32([[cols * 0.2, rows * 0.1], [cols * 0.9, rows * 0.2], [cols * 0.1, rows * 0.9]])

        M = cv2.getAffineTransform(pts1, pts2)
        self.image_affine = cv2.warpAffine(self.image, M, (cols, rows))

    def perspective_transform(self):
        def random_warp(img, row, col):
            height, width, channels = self.image.shape

            print(self.image.shape)



            # warp:
            random_margin = 30
            # x1 = random.randint(-random_margin, random_margin)
            # y1 = random.randint(-random_margin, random_margin)
            # x2 = random.randint(width - random_margin - 1, width - 1)
            # y2 = random.randint(-random_margin, random_margin)
            # x3 = random.randint(width - random_margin - 1, width - 1)
            # y3 = random.randint(height - random_margin - 1, height - 1)
            # x4 = random.randint(-random_margin, random_margin)
            # y4 = random.randint(height - random_margin - 1, height - 1)
            #
            #
            # dx1 = random.randint(-random_margin, random_margin)
            # dy1 = random.randint(-random_margin, random_margin)
            # dx2 = random.randint(width - random_margin - 1, width - 1)
            # dy2 = random.randint(-random_margin, random_margin)
            # dx3 = random.randint(width - random_margin - 1, width - 1)
            # dy3 = random.randint(height - random_margin - 1, height - 1)
            # dx4 = random.randint(-random_margin, random_margin)
            # dy4 = random.randint(height - random_margin - 1, height - 1)

            # c1=[112,667,192,302,670,670,437,437]
            # c2=[100,500,100,500,670,670,300,300]
            #
            # x1,x2,x3,x4,y1,y2,y3,y4=[int(x*702/876) for x in c1]
            # dx1, dx2, dx3, dx4, dy1, dy2, dy3, dy4 = [int(x*702/876) for x in c2]

            x1, x2, x3, x4, y1, y2, y3, y4 =100,400,200,300,333,333,160,160
            dx1, dx2, dx3, dx4, dy1, dy2, dy3, dy4=100,400,100,400,333,333,0,0



            pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
            M_warp = cv2.getPerspectiveTransform(pts1, pts2)
            img_warp = cv2.warpPerspective(img, M_warp, (width, height))
            return M_warp, img_warp

        M_warp, self.image_warp = random_warp(self.image, self.image.shape[0], self.image.shape[1])


    def showPicture(self):
        cv2.imshow('dragon-mother', self.image)
        cv2.imshow('dragon-rotate', self.image_warp)
        # cv2.imshow('dragon-mother-rand', self.imageRand)
        # cv2.imshow('dragon-mother-brighter',self.image_brighter)

        key = cv2.waitKey()
        if key == 27:
            cv2.destroyAllWindows()

    def light_random(self):
        R_rand=random.randint(-50,50)
        if R_rand>0:
            lim=255-R_rand
            self.imageR[self.imageR > lim]=255
            self.imageR[self.imageR <= lim]=self.imageR[self.imageR <= lim]+R_rand
            self.imageR.astype(np.uint8)

        if R_rand<=0:
            lim=0-R_rand
            self.imageR[self.imageR < lim] = 0
            self.imageR[self.imageR >= lim]=self.imageR[self.imageR >= lim]+R_rand
            self.imageR.astype(np.uint8)

        G_rand = random.randint(-50, 50)
        if G_rand > 0:
            lim = 255 - G_rand
            self.imageG[self.imageG > lim] = 255
            self.imageG[self.imageG <= lim] = self.imageG[self.imageG <= lim] + G_rand
            self.imageG.astype(np.uint8)

        if G_rand <=0:
            lim = 0 - G_rand
            self.imageG[self.imageG < lim] = 0
            self.imageG[self.imageG >= lim] = self.imageG[self.imageG >=lim] +G_rand
            self.imageG.astype(np.uint8)

        B_rand = random.randint(-50, 50)
        if B_rand > 0:
            lim = 255 - B_rand
            self.imageB[self.imageB > lim] = 255
            self.imageB[self.imageB <= lim] = self.imageB[self.imageB <= lim] + B_rand
            self.imageB.astype(np.uint8)

        if B_rand <=0:
            lim = 0 - B_rand
            self.imageB[self.imageB < lim] = 0
            self.imageB[self.imageB >= lim] = self.imageB[self.imageB >= lim] +B_rand
            self.imageB.astype(np.uint8)

        self.imageRand=cv2.merge((self.imageR,self.imageG,self.imageB))


    def gamma_correct(self,gamma=1.0):

        def adjust(data,gamma):
            return (data/255)**(1/gamma)*255

        lookup=[]
        for i in range(256):
            lookup.append(adjust(i,gamma))

        lookup=np.array(lookup).astype(np.uint8)

        self.image_brighter=cv2.LUT(self.image,lookup)

        # self.image_brighter=np.empty(self.image.shape,dtype=np.uint8)
        # shape_lis=self.image.shape
        # for i in range(shape_lis[0]):
        #     for j in range(shape_lis[1]):
        #         for k in range(shape_lis[2]):
        #             self.image_brighter[i,j,k]=adjust(self.image[i,j,k],gamma)







if __name__=='__main__':
    filename=r'D:\groceries\AI\CV-CNN\Lesson1\img\mountain.jpg'
    test=imagePpcess(filename)
    test.crop_pic()
    # test.rotate_pic()
    # test.split_pic()
    # test.light_random()
    # test.gamma_correct(1.5)
    # test.affine_transform()
    test.perspective_transform()

    test.showPicture()



