import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

def augment_data(source_dir, target_dir, augmentations, num_augmented_images=3):
    """
    Augment images in the source directory and save them to the target directory.
    
    Parameters:
        source_dir (str): Path to the directory containing original images.
        target_dir (str): Path to the directory where augmented images will be saved.
        augmentations (dict): Augmentation parameters for ImageDataGenerator.
        num_augmented_images (int): Number of augmented images to generate per original image.
    """
    # Initialize the ImageDataGenerator with the specified augmentations
    datagen = ImageDataGenerator(**augmentations)

    # List all image files in the source directory
    images = [img for img in os.listdir(source_dir) if img.endswith(('png', 'jpg', 'jpeg'))]
    print(f"Found {len(images)} images in {source_dir}: {images}")

    for img_name in images:
        img_path = os.path.join(source_dir, img_name)
        img = load_img(img_path)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        # Generate and save augmented images
        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=target_dir, save_prefix='aug', save_format='jpeg'):
            i += 1
            if i >= num_augmented_images:
                break

# Define augmentation parameters
augmentations = {
    'vertical_flip': True,
    'zca_whitening': True,
    'brightness_range': (0.5, 1.5),
}

# Define source and target directories
source_dir = 'classification_dataset/violet'
target_dir = 'bal_classification_dataset/violet'
os.makedirs(target_dir, exist_ok=True)

# Augment the images
augment_data(source_dir, target_dir, augmentations)
