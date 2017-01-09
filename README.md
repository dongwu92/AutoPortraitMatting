# AutoPortraitMatting
Tensorflow implementation of Automatic Portrait Matting on paper "Automatic Portrait Segmentation for Image Stylization"

## Requirement

* Tensorflow
* scipy
* six
* PIL
* numpy

## Data

mkdir Model_zoo and download http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat

download data from http://xiaoyongshen.me/webpages/webpage_automatting/

mkdir data and copy the following folders and files into data/:

  * images_mask/
  * images_tracker/
  * portraitFCN_data/
  * portraitFCN+_data/
  * meanmask.png
  * testlist.mat
  * trainlist.mat

## How to run

FCN.py and FCN_puls.py implements the two models in that paper.
To train:

`
  if __name__ == "__main__":
      tf.app.run()
      #pred()
`

After train, predict:

`
  if __name__ == "__main__":
      #tf.app.run()
      pred()
`

## Ackownledge

Thanks for Xiaoyong Shen and its paper

"Deep Automatic Portrait Matting" Xiaoyong Shen, Xin Tao, Hongyun Gao, Chao Zhou, Jiaya Jia. European Conference on Computer Vision (ECCV), 2016

I can't reach the performance in this paper. So please modify it as you want and inform me if any improvement.
