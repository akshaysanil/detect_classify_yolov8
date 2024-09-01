import os
import shutil
from collections import Counter
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image

def balance_dataset(input_dir, output_dir, target_count, target_size=(224, 224)):
    # Create ImageDataGenerator for augmentation
    datagen = ImageDataGenerator(
        # rotation_range=20,
        # width_shift_range=0.2,
        zca_whitening=True,
        vertical_flip=True,
        brightness_range=(0.5, 1.5),
    )

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Count images in each class
    class_counts = Counter()
    for class_name in os.listdir(input_dir):
        class_dir = os.path.join(input_dir, class_name)
        if os.path.isdir(class_dir):
            class_counts[class_name] = len(os.listdir(class_dir))

    # Process each class
    for class_name, count in class_counts.items():
        class_input_dir = os.path.join(input_dir, class_name)
        class_output_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True)

        # If class has fewer images than target, augment
        if count < target_count:
            images = []
            for img_name in os.listdir(class_input_dir):
                img_path = os.path.join(class_input_dir, img_name)
                img = Image.open(img_path).convert('RGB')
                # img = img.resize(target_size)
                img_array = np.array(img)
                images.append(img_array)

            images = np.array(images)
            augmented_images = datagen.flow(
                images, 
                batch_size=1,
                save_to_dir=class_output_dir,
                save_prefix='aug',
                save_format='png'
            )

            # Generate augmented images
            for i in range(target_count - count):
                next(augmented_images)

        # If class has more images than target, randomly select
        elif count > target_count:
            all_images = os.listdir(class_input_dir)
            selected_images = np.random.choice(all_images, target_count, replace=False)
            for img_name in selected_images:
                src = os.path.join(class_input_dir, img_name)
                dst = os.path.join(class_output_dir, img_name)
                img = Image.open(src).convert('RGB')
                # img = img.resize(target_size)
                img.save(dst)

        # If class has exactly target_count images, copy all
        else:
            for img_name in os.listdir(class_input_dir):
                src = os.path.join(class_input_dir, img_name)
                dst = os.path.join(class_output_dir, img_name)
                img = Image.open(src).convert('RGB')
                # img = img.resize(target_size)
                img.save(dst)

    print("Dataset balancing complete.")

# Usage
input_directory = 'classification_dataset'
output_directory = 'bal_classification_dataset'
target_count = 273  # Set this to your desired number of images per class
target_size = (224, 224)  # Set this to your desired image size

balance_dataset(input_directory, output_directory, target_count, target_size)