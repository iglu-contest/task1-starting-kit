import os
from PIL import Image

def find_file(dir_name):
    for subdir, dirs, files in os.walk(dir_name):
        for file in files:
            #print os.path.join(subdir, file)
            filepath = subdir + os.sep + file
            
            if filepath.endswith(".png"):
                image = Image.open(filepath)
                # print(image.size)
                image.thumbnail((256,256))
                new_dir = subdir.replace("IMAGE_FOLDER", "screenshots")
                # "IMAGE_FOLDER"  is the name of the downloaded folder
                if not os.path.exists(new_dir):
                    os.makedirs(new_dir)
                # new_file_path = (new_dir + os.sep + file).replace('.png', '.jpg')
                new_file_path = (new_dir + os.sep + file)
                image.save(new_file_path, format='JPEG')
                print(filepath)
                print(new_file_path)
            # print("-------------")
            # print(filepath)
            # print(subdir, dirs, files)

path_1 = "/ABC/DEF/IMAGE_FOLDER"
# replace path_1 with the path to the downloaded image folder; 
find_file(path_1)




