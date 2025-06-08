import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def treat_data_set(path, shuffle_value):

    IMG_SIZE = (150, 150)  
    data_dir = path  

    train_datagen = ImageDataGenerator(
        rescale=1./255,                 # Normalização
        rotation_range=40,             # Rotação aleatória
        width_shift_range=0.2,         # Deslocamento horizontal
        height_shift_range=0.2,        # Deslocamento vertical
        shear_range=0.2,               # Cisalhamento
        zoom_range=0.2,                # Zoom aleatório
        horizontal_flip=True,          # Inversão horizontal
        fill_mode='nearest'            # Preenchimento de pixels faltantes
    )

    # Geração dos dados a partir das pastas
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=32,
        class_mode='binary',  # ou 'categorical' se tiver mais de duas classes
        shuffle= shuffle_value
    )

    return train_generator
