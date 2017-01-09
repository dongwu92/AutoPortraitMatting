import numpy as np
import scipy.io as sio
import os
from PIL import Image
import math
from scipy import misc

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

    def __init__(self, imgs_path, batch_size=1):
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
            rimat = np.zeros((self.batch_size, self.img_height, self.img_width, 6), dtype=np.float)
            ramat = np.zeros((self.batch_size, self.img_height, self.img_width, 1), dtype=np.int)
            self.cur_batch += 1 # output a new batch
            for i in range(self.batch_size):
                rimat[i] = self.cur_imgs.pop(0)
                ramat[i, :, :, 0] = self.cur_labels.pop(0)
            #print('batch:', self.cur_batch, 'at img:', self.imgs[self.cur_ind], 'generate image shape', rimat.shape, 'and label shape', ramat.shape)
            return rimat, ramat
        return [], []

    def rotateNormalizedCord(self, matx, maty, angle):
        h, w = matx.shape
        x_avg = np.mean(matx)
        x_min = np.min(matx)
        y_avg = np.mean(maty)
        y_min = np.min(maty)
        xmat = np.zeros((h, w), dtype=np.float)
        ymat = np.zeros((h, w), dtype=np.float)
        for k in range(h):
            for j in range(w):
                cor_y = k - h / 2
                cor_x = j - w / 2
                if cor_x == 0 and cor_y == 0:
                    xmat[k][j] = x_avg
                    ymat[k][j] = y_avg
                else:
                    x_dis = math.cos(math.pi / 2 - angle) * (-math.tan(math.pi / 2 - angle) * cor_x + cor_y)
                    xmat[k][j] = x_avg - (x_avg - x_min) * x_dis * 2 / w
                    y_dis = math.cos(angle) * (math.tan(angle) * cor_x + cor_y)
                    ymat[k][j] = y_avg + (y_avg - y_min) * y_dis * 2 / h
        return xmat, ymat

    def scaleNormalizedCord(self, matx, maty, scale):
        h, w = matx.shape
        x_avg = np.mean(matx)
        x_max = np.max(matx)
        y_avg = np.mean(maty)
        y_max = np.max(maty)
        xmat = np.zeros((h, w), dtype=np.float)
        ymat = np.zeros((h, w), dtype=np.float)
        for k in range(h):
            for j in range(w):
                cor_y = k - h / 2
                cor_x = j - w / 2
                xmat[k][j] = x_avg + (x_max - x_avg) * cor_x / scale
                ymat[k][j] = y_avg + (y_max - y_avg) * cor_y / scale
        return xmat, ymat

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
        img_path = 'data/portraitFCN+_data/' + stp + '.mat'
        alpha_path = 'data/images_mask/' + stp + '_mask.mat'
        if os.path.exists(img_path) and os.path.exists(alpha_path):
            imat = sio.loadmat(img_path)['img']
            amat = sio.loadmat(alpha_path)['mask']
            nimat = np.array(imat, dtype=np.float)
            namat = np.array(amat, dtype=np.int)
            imgs.append(nimat)
            labels.append(namat)

            angs = [-45, -22, 22, 45]
            gammas = [0.5, 0.8, 1.2, 1.5]
            scales = [0.6, 0.8, 1.2, 1.5]
            h, w, _ = nimat.shape
            org_mat = np.zeros((h, w, 3), dtype=np.int)
            app_mat = np.zeros((h, w, 3), dtype=np.int)
            min3 = np.min(nimat[:, :, 3])
            min4 = np.min(nimat[:, :, 4])
            min5 = np.min(nimat[:, :, 5])
            ran3 = np.max(nimat[:, :, 3]) - min3
            ran4 = np.max(nimat[:, :, 4]) - min4
            ran5 = np.max(nimat[:, :, 5]) - min5
            org_mat[:, :, 0] = np.round(nimat[:, :, 2] * 255 + 122.675)
            org_mat[:, :, 1] = np.round(nimat[:, :, 1] * 255 + 116.669)
            org_mat[:, :, 2] = np.round(nimat[:, :, 0] * 255 + 104.008)
            if ran3 != 0:
                app_mat[:, :, 0] = np.round((nimat[:, :, 3] - min3) * 255 / ran3)
            else:
                app_mat[:, :, 0] = min3
            if ran4 != 0:
                app_mat[:, :, 1] = np.round((nimat[:, :, 4] - min4) * 255 / ran4)
            else:
                app_mat[:, :, 0] = min4
            if ran5 != 0:
                app_mat[:, :, 2] = np.round((nimat[:, :, 5] - min5) * 255 / ran5)
            else:
                app_mat[:, :, 0] = min5
            i_img = Image.fromarray(np.uint8(org_mat))
            p_img = Image.fromarray(np.uint8(app_mat))
            a_img = Image.fromarray(np.uint8(amat))
            for i in range(4):
                # rotation
                tmpi_img = i_img.rotate(angs[i])
                tmpp_img = p_img.rotate(angs[i])
                tmpa_img = a_img.rotate(angs[i])
                tmpri_img = np.array(tmpi_img, dtype=np.int)
                tmprp_img = np.array(tmpp_img, dtype=np.int)
                rot_p1, rot_p2 = self.rotateNormalizedCord(nimat[:, :, 3], nimat[:, :, 4], angs[i] * math.pi / 180)
                rimat = np.zeros((h, w, 6), dtype=np.float)
                rimat[:, :, 0] = (tmpri_img[:, :, 2] - 104.008) / 255
                rimat[:, :, 1] = (tmpri_img[:, :, 1] - 116.669) / 255
                rimat[:, :, 2] = (tmpri_img[:, :, 0] - 122.675) / 255
                rimat[:, :, 3] = rot_p1
                rimat[:, :, 4] = rot_p2
                rimat[:, :, 5] = tmprp_img[:, :, 2] * ran5 / 255 + min5
                imgs.append(rimat)
                labels.append(np.array(tmpa_img, dtype=np.int))
                # gamma transformation
                tmp_nimat = np.array(imat, dtype=np.float)
                tmp_nimat[:, :, 0] = tmp_nimat[:, :, 0] + 104.008 / 255
                tmp_nimat[:, :, 0] = (pow(tmp_nimat[:, :, 0], gammas[i]) - pow(104.008 / 255, gammas[i]))
                tmp_nimat[:, :, 1] = tmp_nimat[:, :, 1] + 116.669 / 255
                tmp_nimat[:, :, 1] = (pow(tmp_nimat[:, :, 1], gammas[i]) - pow(116.669 / 255, gammas[i]))
                tmp_nimat[:, :, 2] = tmp_nimat[:, :, 2] + 122.675 / 255
                tmp_nimat[:, :, 2] = (pow(tmp_nimat[:, :, 2], gammas[i]) - pow(122.675 / 255, gammas[i]))
                imgs.append(tmp_nimat)
                labels.append(namat)
                # scale transformation
                if scales[i] > 1.0:
                    resize_box = (round(scales[i] * w), round(scales[i] * h))
                    si_img = i_img.resize(resize_box, Image.ANTIALIAS)
                    sp_img = p_img.resize(resize_box, Image.ANTIALIAS)
                    sa_img = a_img.resize(resize_box, Image.ANTIALIAS)
                    crop_up, crop_down = (scales[i] - 1) / 2, (scales[i] + 1) / 2
                    crop_box = (round(crop_up * w), round(crop_up * h), round(crop_down * w), round(crop_down * h))
                    ci_img = si_img.crop(crop_box)
                    cp_img = sp_img.crop(crop_box)
                    ca_img = sa_img.crop(crop_box)
                    tmpsi_img = np.array(ci_img, dtype=np.int)
                    tmpsp_img = np.array(cp_img, dtype=np.int)
                    tmpsa_img = np.array(ca_img, dtype=np.int)
                    simat = np.zeros(imat.shape, dtype=np.float)
                    simat[:, :, 0] = (tmpsi_img[:, :, 2] - 104.008) / 255
                    simat[:, :, 1] = (tmpsi_img[:, :, 1] - 116.669) / 255
                    simat[:, :, 2] = (tmpsi_img[:, :, 0] - 122.675) / 255
                    xmat, ymat = self.scaleNormalizedCord(nimat[:, :, 3], nimat[:, :, 4], scales[i] * 300)
                    simat[:, :, 3] = xmat
                    simat[:, :, 4] = ymat
                    simat[:, :, 5] = tmpsp_img[:, :, 2] * ran5 / 255 + min5
                    imgs.append(simat)
                    labels.append(tmpsa_img)
                else:
                    resize_box = (round(scales[i] * w), round(scales[i] * h))
                    si_img = i_img.resize(resize_box, Image.ANTIALIAS)
                    sp_img = p_img.resize(resize_box, Image.ANTIALIAS)
                    sa_img = a_img.resize(resize_box, Image.ANTIALIAS)
                    tmpsi_img = np.array(si_img, dtype=np.int)
                    tmpsp_img = np.array(sp_img, dtype=np.int)
                    tmpsa_img = np.array(sa_img, dtype=np.int)
                    simat = np.zeros(imat.shape, dtype=np.float)
                    samat = np.zeros(amat.shape, dtype=np.int)
                    crop_up, crop_down = (1 - scales[i]) / 2, (1 + scales[i]) / 2
                    simat[round(crop_up * h):round(crop_down * h), round(crop_up * w):round(crop_down * w), 0] = (tmpsi_img[:, :, 2] - 104.008) / 255
                    simat[round(crop_up * h):round(crop_down * h), round(crop_up * w):round(crop_down * w), 1] = (tmpsi_img[:, :, 1] - 116.669) / 255
                    simat[round(crop_up * h):round(crop_down * h), round(crop_up * w):round(crop_down * w), 2] = (tmpsi_img[:, :, 0] - 122.675) / 255
                    simat[round(crop_up * h):round(crop_down * h), round(crop_up * w):round(crop_down * w), 5] = tmpsp_img[:, :, 2] * ran5 / 255 + min5
                    samat[round(crop_up * h):round(crop_down * h), round(crop_up * w):round(crop_down * w)] = tmpsa_img
                    xmat, ymat = self.scaleNormalizedCord(nimat[:, :, 3], nimat[:, :, 4], scales[i] * 300)
                    simat[:, :, 3] = xmat
                    simat[:, :, 4] = ymat
                    imgs.append(simat)
                    labels.append(samat)
        return imgs, labels


class TestDataset:
    imgs = []
    max_batch = 0
    batch_size = 0
    cur_batch = 0 # index of batch generated
    cur_ind = -1 # index of current image in imgs
    img_width = 600
    img_height = 800

    def __init__(self, imgs_path, batch_size=1):
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
            rimat = np.zeros((self.batch_size, self.img_height, self.img_width, 6), dtype=np.float)
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
        img_path = 'data/portraitFCN+_data/' + stp + '.mat'
        alpha_path = 'data/images_mask/' + stp + '_mask.mat'
        if os.path.exists(img_path) and os.path.exists(alpha_path):
            imat = sio.loadmat(img_path)['img']
            amat = sio.loadmat(alpha_path)['mask']
            nimat = np.array(imat, dtype=np.float)
            namat = np.array(amat, dtype=np.int)
            h, w, _ = nimat.shape
            org_mat = np.zeros((h, w, 3), dtype=np.int)
            for i in range(h):
                for j in range(w):
                    org_mat[i][j][0] = round(nimat[i][j][2] * 255 + 122.675)
                    org_mat[i][j][1] = round(nimat[i][j][1] * 255 + 116.669)
                    org_mat[i][j][2] = round(nimat[i][j][0] * 255 + 104.008)
            return nimat, namat, org_mat
        return None, None, None


if __name__ == '__main__':
    dat = BatchDatset('data/trainlist.mat', batch_size=13)
    rimat, ramat = dat.next_batch()
    for i in range(13):
        imat = rimat[i]
        amat = ramat[i]
        rgb = np.zeros((imat.shape[0], imat.shape[1], 3), dtype=np.int)
        rgb[:, :, 0] = np.round(imat[:, :, 2] * 255 + 122.675)
        rgb[:, :, 1] = np.round(imat[:, :, 1] * 255 + 116.669)
        rgb[:, :, 2] = np.round(imat[:, :, 0] * 255 + 104.008)
        misc.imsave('res/org' + str(i) + '.jpg', rgb)
        xxc = imat[:, :, 3]
        xxc = np.round((xxc - np.min(xxc) / (np.max(xxc) - np.min(xxc))) * 255)
        misc.imsave('res/xxc' + str(i) + '.jpg', xxc)
        yyc = imat[:, :, 4]
        yyc = np.round((yyc - np.min(yyc) / (np.max(yyc) - np.min(yyc))) * 255)
        misc.imsave('res/yyc' + str(i) + '.jpg', yyc)
        mean = imat[:, :, 5] * 255
        misc.imsave('res/mean' + str(i) + '.jpg', mean)
        alpha = amat * 255
        alpha = alpha.reshape((alpha.shape[0], alpha.shape[1]))
        misc.imsave('res/alpha' + str(i) + '.jpg', alpha)
