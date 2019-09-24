import glob
import os

gif_name = 'riskRatio_2018'
images_path = '/Users/jecv/Documents/Uniandes/Maestria/Proyecto_Ibague/DATA/Figures/2018/'

file_list = glob.glob(images_path+'/'+'*.png') # Get all the pngs in the current directory

list.sort(file_list, key=lambda x: int(x.split('_')[3])) # Sort the images by # given by epidemic week

with open(images_path+'image_list.txt', 'w') as file:
    for item in file_list:
        file.write("%s\n" % item)

os.system('convert -delay 60 @image_list.txt {}.gif'.format(gif_name)) # On windows convert is 'magick'
####    convert -delay 40 @image_list.txt cases_week_2019.gif    ####ipython
