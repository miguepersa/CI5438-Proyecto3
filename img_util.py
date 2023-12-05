from PIL import Image
import pandas as pd
import cv2
import numpy as np
from K_means import *

def segmentacion_imagen(input_file: str, output_file:str, colores: int, iteraciones: int):
    df_image, width, height = read_picture_rgb(f'./input/{input_file}')
    km = Kmeans(df_image, colores)
    km.train(iteraciones)
    out = update_picture_w_centroids(km.X, km.centroides)
    write_picture_from_rgb(output_file, out, width, height)

def read_picture_rgb(picture_name):
    img = cv2.imread(picture_name)
    height = len(img)
    width = len(img[0])
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    arr = []

    for row in image_rgb:
        for pixel in row:
            arr.append(list(pixel))

    dictionary = {
        "r" : [pixel[0] for pixel in arr],
        "g" : [pixel[1] for pixel in arr],
        "b" : [pixel[2] for pixel in arr]
    }

    df = pd.DataFrame(dictionary)
    return df, width, height

def update_picture_w_centroids(data: pd.DataFrame, centroides):
    out = []
    for _, row in data.iterrows():
        c = int(row["cluster"])
        out.append(centroides[c])

    return out

def write_picture_from_rgb(out_name, picture_array, width, height):
    out = []
    
    for i in range(height):
        row = []
        for j in range(width):
            row.append(np.array(picture_array[(i*width)+j]))
        
        out.append(np.array(row))

    out = np.array(out)
    img = Image.fromarray(out.astype('uint8'))
    img.save(f"./output/{out_name}")