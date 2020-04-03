from pathlib import Path
import requests
from PIL import Image
from numpy import asarray, shape


class Image_Loader:

    def __init__(self, frog_dir="data/frog_images/", not_frog_dir="data/not_frog_images/"):
        self.frog_dir = Path(frog_dir)
        self.not_frog_dir = Path(not_frog_dir)

    # generate non-frog images
    def generate_random_images(self):
        for i in range(int(len(list(self.frog_dir.iterdir()))/2)):
            url = "https://picsum.photos/64/"
            response = requests.get(url)
            if response.status_code == 200:
                filename = "not_frog_{}.png".format(i)
                filepath = self.not_frog_dir.joinpath(filename)
                filepath.write_bytes(response.content)

    #  paste frog images onto blank RGB background to remove alpha channel
    def remove_alpha(self, dir="data/frog_images/"):
        filepath = Path(dir).joinpath("no_alpha/")
        if filepath.is_dir() == 0:
            Path.mkdir(filepath)

        for file in self.frog_dir.iterdir():
            if file.suffix == ".png":
                file_split = str(file).split('/')
                filename = file_split[2]
                # create 3-channel version of frog image
                image = Image.open(file)
                background = Image.new("RGB", image.size, (255,255,255))
                background.paste(image, (0,0), image)
                background.save(filepath.joinpath(filename), "PNG")

    # load frog and non-frog images into numpy arrays
    def load_images(self, frog_dir="data/frog_images/no_alpha", not_frog_dir="data/not_frog_images/"):
        images, labels = list(), list()
        for filename in Path(frog_dir).iterdir():
            label = 1
            if filename.suffix == ".png":
                image = Image.open(filename)
                # turn image into numpy array
                data = asarray(image)
                images.append(data)
                labels.append(label)
        for filename in Path(not_frog_dir).iterdir():
            label = 0
            if filename.suffix == ".png":
                image = Image.open(filename)
                # turn image into numpy array
                data = asarray(image)
                images.append(data)
                labels.append(label)
        images = asarray(images)
        labels = asarray(labels)
        return [images, labels]
