import os
import numpy as np
import csv
from skimage import io
from skimage.io import imread
from skimage.color import rgb2gray
from PIL import Image
import argparse
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio
import math

# this function is for read image,the input is directory name
def process_directory(directory_name, subdirname):
    for filename in os.listdir(r"./"+directory_name):
        if filename.endswith('B.tif'):
            img = imread(directory_name + "/" + filename)
            composite = tif_composite(img)
            outimgdir = './composite_rgb'
            if not os.path.exists(outimgdir + '/' + subdirname):
                os.mkdir(outimgdir + '/' + subdirname)
            composite.save(os.path.join(outimgdir, subdirname, filename))


def tif_composite(img):
    img = ((img + 1.0) / 2.0) * 255

    a = img[:, :, 0] #dapi
    b = img[:, :, 1] #CD3
    c = img[:, :, 2] #Panck
    rgb_img = np.zeros((1024, 1024, 3), 'uint8')
    rgb_img[:, :, 0] = c
    rgb_img[:, :, 1] = b
    rgb_img[:, :, 2] = a
    rgb_img = Image.fromarray(rgb_img)

    return(rgb_img)

def compute_ssim(directory_name):
    csv_path = os.path.join(directory_name, 'score.csv')
    f = open(csv_path, 'w', encoding='utf-8',newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(['file_name', 'dapi', 'cd3', 'panck','average'])
    for filename in os.listdir(r"./" + directory_name):
        if filename.endswith('_fake_B.tif'):
            fake_mihc = imread(directory_name + "/" + filename)
            nosuff_name = filename[0:-11]
            real_mihc_name = filename[0:-10]+'real_B.tif'
            real_mihc = imread(directory_name + '/' + real_mihc_name)
            real_dapi =real_mihc[:, :, 0]
            real_cd3 = real_mihc[:, :, 1]
            real_panck = real_mihc[:, :, 2]

            fake_dapi = fake_mihc[:, :, 0]
            fake_cd3 = fake_mihc[:, :, 1]
            fake_panck = fake_mihc[:, :, 2]
            dapi_score = ssim(real_dapi, fake_dapi)
            cd3_score = ssim(real_cd3, fake_cd3)
            panck_score = ssim(real_panck, fake_panck)
            average_score = np.average([dapi_score,cd3_score,panck_score])
            csv_writer.writerow([nosuff_name,dapi_score,cd3_score,panck_score,average_score])
    f.close()

def compute_metrics(directory_name):
    csv_path = os.path.join(directory_name, 'score.csv')
    f = open(csv_path, 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(['file_name', 'dapi', 'cd3', 'panck', 'average_ssim','dapi_pr','cd3_pr','panck_pr',
                         'average_pearson', 'dapi_psnr', 'cd3_psnr', 'panck_psnr', 'average_psnr'])
    for filename in os.listdir(r"./" + directory_name):
        if filename.endswith('_fake_B.tif'):
            fake_mihc = imread(directory_name + "/" + filename)
            nosuff_name = filename[0:-11]
            real_mihc_name = filename[0:-10] + 'real_B.tif'
            real_mihc = imread(directory_name + '/' + real_mihc_name)
            real_dapi = real_mihc[:, :, 0]
            real_cd3 = real_mihc[:, :, 1]
            real_panck = real_mihc[:, :, 2]

            fake_dapi = fake_mihc[:, :, 0]
            fake_cd3 = fake_mihc[:, :, 1]
            fake_panck = fake_mihc[:, :, 2]

            dapi_score = ssim(real_dapi, fake_dapi, data_range=255, multichannel=True)
            cd3_score = ssim(real_cd3, fake_cd3, data_range=255, multichannel=True)
            panck_score = ssim(real_panck, fake_panck, data_range=255, multichannel=True)
            average_score = np.average([dapi_score, cd3_score, panck_score])

            tiny = 1e-15
            dapi_f = np.array(fake_dapi).flatten().astype(float)
            dapi_f[0] = dapi_f[0] + tiny
            dapi_r = np.array(real_dapi).flatten().astype(float)
            dapi_r[0] = dapi_r[0] + tiny
            cd3_f = np.array(fake_cd3).flatten().astype(float)
            cd3_f[0] = cd3_f[0] + tiny
            cd3_r = np.array(real_cd3).flatten().astype(float)
            cd3_r[0] = cd3_r[0] + tiny
            panck_f = np.array(fake_panck).flatten().astype(float)
            panck_f[0] = panck_f[0] + tiny
            panck_r = np.array(real_panck).flatten().astype(float)
            panck_r[0] = panck_r[0] + tiny
            corr_dapi = np.corrcoef(dapi_f, dapi_r)
            corr_dapi = corr_dapi[0, 1]
            corr_cd3 = np.corrcoef(cd3_f, cd3_r)
            corr_cd3 = corr_cd3[0, 1]
            corr_panck = np.corrcoef(panck_f, panck_r)
            corr_panck = corr_panck[0, 1]
            corr_average = (corr_dapi + corr_cd3 + corr_panck) / 3

            psnr_dapi = peak_signal_noise_ratio(fake_dapi, real_dapi)
            psnr_cd3 = peak_signal_noise_ratio(fake_cd3, real_cd3)
            psnr_panck = peak_signal_noise_ratio(fake_panck, real_panck)

            psnr_average = (psnr_dapi + psnr_cd3 + psnr_panck) / 3

            csv_writer.writerow([nosuff_name, dapi_score, cd3_score, panck_score, average_score, corr_dapi,
                                 corr_cd3, corr_panck, corr_average, psnr_dapi, psnr_cd3, psnr_panck,
                                 psnr_average])
    f.close()


def compute_dapi_ssim(directory_name):
    csv_path = os.path.join(directory_name, 'score.csv')
    #csv_path = csv_path.replace('\\', '/')
    f = open(csv_path, 'w', encoding='utf-8',newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(['file_name', 'dapi'])
    for filename in os.listdir(r"./" + directory_name):
        if filename.endswith('_fake_B.tif'):
            fake_mihc = imread(directory_name + "/" + filename)
            nosuff_name = filename[0:-11]
            real_mihc_name = filename[0:-10]+'real_B.tif'
            #real_mihc_name = filename[0:-11] + '.tif'
            real_mihc = imread(directory_name + '/' + real_mihc_name)
            real_dapi =rgb2gray(real_mihc)
            fake_dapi = rgb2gray(fake_mihc)
            print(real_dapi.shape)
            print(fake_dapi.shape)
            dapi_score = ssim(real_dapi, fake_dapi)
            csv_writer.writerow([nosuff_name, dapi_score])
    f.close()

def compute_cd3_ssim(directory_name):
    csv_path = os.path.join(directory_name, 'score.csv')
    #csv_path = csv_path.replace('\\', '/')
    f = open(csv_path, 'w', encoding='utf-8',newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(['file_name', 'cd3'])
    for filename in os.listdir(r"./" + directory_name):
        if filename.endswith('_fake_B.tif'):
            fake_mihc = imread(directory_name + "/" + filename)
            nosuff_name = filename[0:-11]
            real_mihc_name = filename[0:-10]+'real_B.tif'
            #real_mihc_name = filename[0:-11] + '.tif'
            real_mihc = imread(directory_name + '/' + real_mihc_name)
            real_cd3 = rgb2gray(real_mihc)
            fake_cd3 = rgb2gray(fake_mihc)
            cd3_score = ssim(real_cd3, fake_cd3)
            csv_writer.writerow([nosuff_name,cd3_score])
    f.close()

def compute_panck_ssim(directory_name):
    csv_path = os.path.join(directory_name, 'score.csv')
    #csv_path = csv_path.replace('\\', '/')
    f = open(csv_path, 'w', encoding='utf-8',newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(['file_name', 'panck'])
    for filename in os.listdir(r"./" + directory_name):
        if filename.endswith('_fake_B.tif'):
            fake_mihc = imread(directory_name + "/" + filename)
            nosuff_name = filename[0:-11]
            real_mihc_name = filename[0:-10]+'real_B.tif'
            #real_mihc_name = filename[0:-11] + '.tif'
            real_mihc = imread(directory_name + '/' + real_mihc_name)
            real_panck = rgb2gray(real_mihc)
            fake_panck = rgb2gray(fake_mihc)
            panck_score = ssim(real_panck, fake_panck)
            csv_writer.writerow([nosuff_name,panck_score])
    f.close()

def validation_train(real_mihc, fake_mihc):
    real_mihc  = (real_mihc + 1.0) / 2.0
    fake_mihc = (fake_mihc + 1.0) / 2.0

    real_mihc = (real_mihc.cpu().detach().numpy() * 255).astype(np.uint8)
    fake_mihc = (fake_mihc.cpu().detach().numpy() * 255).astype(np.uint8)

    real_mihc = np.squeeze(real_mihc).transpose([2, 1, 0])
    fake_mihc = np.squeeze(fake_mihc).transpose([2, 1, 0])

    real_dapi = real_mihc[:, :, 0]
    real_cd3 = real_mihc[:, :, 1]
    real_panck = real_mihc[:, :, 2]

    fake_dapi = fake_mihc[:, :, 0]
    fake_cd3 = fake_mihc[:, :, 1]
    fake_panck = fake_mihc[:, :, 2]

    dapi_score = ssim(real_dapi, fake_dapi, data_range=255, multichannel=True)
    cd3_score = ssim(real_cd3, fake_cd3, data_range=255, multichannel=True)
    panck_score = ssim(real_panck, fake_panck, data_range=255, multichannel=True)
    average_score = np.average([dapi_score, cd3_score, panck_score])

    return dapi_score, cd3_score, panck_score, average_score

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--srcdir", type=str, help="process this directory.")
    parser.add_argument("--outdir", type=str, help="save in this directory.")
    args = parser.parse_args()
    directory_name = args.srcdir
    subdir = args.outdir
    #process_directory(directory_name, subdir)
    #compute_ssim(directory_name)
    compute_metrics(directory_name)


    '''
    directory_name_dapi = 'results/mihc_pix2pix_0103_lr001_cosine_dapi/test_280/images'
    directory_name_cd3 = 'results/mihc_pix2pix_0103_lr001_cosine_cd3/test_280/images'
    directory_name_panck = 'results/mihc_pix2pix_0103_lr001_cosine_panck/test_280/images'
    compute_dapi_ssim(directory_name_dapi)
    compute_cd3_ssim(directory_name_cd3)
    compute_panck_ssim(directory_name_panck)
    '''