import numpy as np
import scipy.io as sio
import os
from PIL import Image

class BatchDatset:
    imgs = []
    max_batch = 0
    batch_size = 0
    cur_imgs = []
    cur_labels = []
    cur_batch = 0 # index of batch generated
    cur_ind = 0 # index of current image in imgs
    img_width = 600
    img_height = 800

    def __init__(self, imgs_path, batch_size=2):
        self.imgs = sio.loadmat(imgs_path)['trainlist'][0]
        #self.labels = sio.loadmat(labels_path)['test_list'][0]
        self.batch_size = batch_size
        #self.max_batch = len(self.imgs) * 9 / batch_size
        self.cur_imgs, self.cur_labels = self.get_variations(self.imgs[0])

    def next_batch(self):
        while len(self.cur_imgs) < self.batch_size: # if not enough, get the next image
            self.cur_ind += 1
            #print('appending', self.cur_ind)
            if self.cur_ind >= len(self.imgs):
                #print('leaving', self.cur_ind)
                break
            cur_name = self.imgs[self.cur_ind]
            tmp_imgs, tmp_labels = self.get_variations(cur_name)
            self.cur_imgs += tmp_imgs
            self.cur_labels += tmp_labels
        if len(self.cur_imgs) >= self.batch_size:
            #print('getting', self.cur_ind)
            rimat = np.zeros((self.batch_size, self.img_height, self.img_width, 3), dtype=np.float)
            ramat = np.zeros((self.batch_size, self.img_height, self.img_width, 1), dtype=np.int)
            self.cur_batch += 1 # output a new batch
            for i in range(self.batch_size):
                rimat[i] = self.cur_imgs.pop(0)
                ramat[i, :, :, 0] = self.cur_labels.pop(0)
            #print('batch:', self.cur_batch, 'at img:', self.imgs[self.cur_ind], 'generate image shape', rimat.shape, 'and label shape', ramat.shape)
            return rimat, ramat
        return [], []

    def get_variations(self, img_name):
        imgs = []
        labels = []
        stp = str(img_name)
        if img_name < 10:
            stp = '0000' + stp
        elif img_name < 100:
            stp = '000' + stp
        elif img_name < 1000:
            stp = '00' + stp
        else:
            stp = '0' + stp
        img_path = 'data/portraitFCN_data/' + stp + '.mat'
        alpha_path = 'data/images_mask/' + stp + '_mask.mat'
        if os.path.exists(img_path) and os.path.exists(alpha_path):
            imat = sio.loadmat(img_path)['img']
            amat = sio.loadmat(alpha_path)['mask']
            nimat = np.array(imat, dtype=np.float)
            namat = np.array(amat, dtype=np.int)
            imgs.append(nimat)
            labels.append(namat)

            angs = [-45, -22, 22, 45]
            gammas = [0.8, 0.9, 1.1, 1.2]
            org_mat = np.zeros(nimat.shape, dtype=np.int)
            h, w, _ = nimat.shape
            for i in range(h):
                for j in range(w):
                    org_mat[i][j][0] = round(nimat[i][j][2] * 255 + 122.675)
                    org_mat[i][j][1] = round(nimat[i][j][1] * 255 + 116.669)
                    org_mat[i][j][2] = round(nimat[i][j][0] * 255 + 104.008)
            i_img = Image.fromarray(np.uint8(org_mat))
            a_img = Image.fromarray(np.uint8(amat))
            for i in range(4):
                tmpi_img = i_img.rotate(angs[i])
                tmpa_img = a_img.rotate(angs[i])
                tmpri_img = np.array(tmpi_img, dtype=np.int)
                rimat = np.zeros(tmpri_img.shape, dtype=np.float)
                for k in range(h):
                    for j in range(w):
                        rimat[k][j][0] = (tmpri_img[k][j][2] * 1.0 - 104.008) / 255
                        rimat[k][j][1] = (tmpri_img[k][j][1] * 1.0 - 116.669) / 255
                        rimat[k][j][2] = (tmpri_img[k][j][0] * 1.0 - 122.675) / 255
                imgs.append(rimat)
                labels.append(np.array(tmpa_img, dtype=np.int))
                tmp_nimat = np.array(imat, dtype=np.float)
                tmp_nimat[:, :, 0] = tmp_nimat[:, :, 0] * 255 + 104.01
                tmp_nimat[:, :, 0] = (pow(tmp_nimat[:, :, 0], gammas[i]) - pow(104.01, gammas[i])) / pow(255, gammas[i])
                tmp_nimat[:, :, 1] = tmp_nimat[:, :, 1] * 255 + 116.67
                tmp_nimat[:, :, 1] = (pow(tmp_nimat[:, :, 1], gammas[i]) - pow(116.67, gammas[i])) / pow(255, gammas[i])
                tmp_nimat[:, :, 2] = tmp_nimat[:, :, 2] * 255 + 122.68
                tmp_nimat[:, :, 2] = (pow(tmp_nimat[:, :, 2], gammas[i]) - pow(122.68, gammas[i])) / pow(255, gammas[i])
                imgs.append(tmp_nimat)
                labels.append(namat)
        return imgs, labels


class TestDataset:
    imgs = []
    max_batch = 0
    batch_size = 0
    cur_batch = 0 # index of batch generated
    cur_ind = -1 # index of current image in imgs
    img_width = 600
    img_height = 800

    def __init__(self, imgs_path, batch_size=2):
        self.imgs = sio.loadmat(imgs_path)['testlist'][0]
        #self.labels = sio.loadmat(labels_path)['test_list'][0]
        self.batch_size = batch_size
        #self.max_batch = len(self.imgs) * 9 / batch_size
        #self.cur_imgs, self.cur_labels = self.get_images(self.imgs[0])

    def next_batch(self):
        cur_imgs = []
        cur_labels = []
        cur_orgs = []
        while len(cur_imgs) < self.batch_size: # if not enough, get the next image
            self.cur_ind += 1
            #print('appending', self.cur_ind)
            if self.cur_ind >= len(self.imgs):
                #print('leaving', self.cur_ind)
                break
            cur_name = self.imgs[self.cur_ind]
            tmp_img, tmp_label, tmp_org = self.get_images(cur_name)
            if tmp_img is not None:
                cur_imgs.append(tmp_img)
                cur_labels.append(tmp_label)
                cur_orgs.append(tmp_org)
        if len(cur_imgs) == self.batch_size:
            #print('getting', self.cur_ind)
            rimat = np.zeros((self.batch_size, self.img_height, self.img_width, 3), dtype=np.float)
            org_mat = np.zeros((self.batch_size, self.img_height, self.img_width, 3), dtype=np.int)
            ramat = np.zeros((self.batch_size, self.img_height, self.img_width, 1), dtype=np.int)
            self.cur_batch += 1 # output a new batch
            for i in range(self.batch_size):
                rimat[i] = cur_imgs.pop(0)
                org_mat[i] = cur_orgs.pop(0)
                ramat[i, :, :, 0] = cur_labels.pop(0)
            #print('getting', ramat[0, 200:210, 200:220])
            #print('batch:', self.cur_batch, 'at img:', self.imgs[self.cur_ind], 'generate image shape', rimat.shape, 'and label shape', ramat.shape)
            return rimat, ramat, org_mat
        return [], [], []

    def get_images(self, img_name):
        stp = str(img_name)
        if img_name < 10:
            stp = '0000' + stp
        elif img_name < 100:
            stp = '000' + stp
        elif img_name < 1000:
            stp = '00' + stp
        else:
            stp = '0' + stp
        img_path = 'data/portraitFCN_data/' + stp + '.mat'
        alpha_path = 'data/images_mask/' + stp + '_mask.mat'
        if os.path.exists(img_path) and os.path.exists(alpha_path):
            imat = sio.loadmat(img_path)['img']
            amat = sio.loadmat(alpha_path)['mask']
            nimat = np.array(imat, dtype=np.float)
            namat = np.array(amat, dtype=np.int)
            org_mat = np.zeros(nimat.shape, dtype=np.int)
            h, w, _ = nimat.shape
            for i in range(h):
                for j in range(w):
                    org_mat[i][j][0] = round(nimat[i][j][2] * 255 + 122.675)
                    org_mat[i][j][1] = round(nimat[i][j][1] * 255 + 116.669)
                    org_mat[i][j][2] = round(nimat[i][j][0] * 255 + 104.008)
            return nimat, namat, org_mat
        return None, None, None

if __name__ == '__main__':
    data = BatchDatset('data/trainlist.mat')
    '''ri, ra = data.next_batch()
    while len(ri) != 0:
        ri, ra = data.next_batch()
        print(np.sum(ra))'''
    imgs, labels = data.get_variations(47)
    cnt = 0
    for img in imgs:
        mat = np.zeros(img.shape, dtype=np.int)
        h, w, _ = img.shape
        for i in range(h):
            for j in range(w):
                mat[i][j][0] = round(img[i][j][2] * 255 + 122.675)
                mat[i][j][1] = round(img[i][j][1] * 255 + 116.669)
                mat[i][j][2] = round(img[i][j][0] * 255 + 104.008)
        im = Image.fromarray(np.uint8(mat))
        im.save('img-'+str(cnt)+'.jpg')
        cnt += 1
