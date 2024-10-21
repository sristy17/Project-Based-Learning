from tensorflow.keras.preprocessing.image import ImageDataGenerator

from skimage import io

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.0,
    height_shift_range=0.0,
    shear_range=0.2,
    zoom_range=-0.6,
    horizontal_flip=False,
    fill_mode='constant'
)

x =  io.imread("ORB_Sristy/input/input_img.jpg")

x = x.reshape((1, ) + x.shape)

i = 0
for batch in datagen.flow(x, batch_size = 16,
                          save_to_dir = "/home/sristy/Desktop/ORB-Descriptor/ORB_Sristy/output",
                          save_prefix='output_img',
                          save_format='jpg'):
    
    i+=1
    if i > 10:
        break