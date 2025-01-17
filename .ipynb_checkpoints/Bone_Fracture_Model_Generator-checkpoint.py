# Basic Libraries
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import json
import traceback as tb
from datetime import datetime as dt
from joblib import dump, load
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,classification_report

################################################################################
#Function to check for corrupted images
################################################################################

def fn_remove_corrupted_images_tf(dataset_dir):
    num_deleted = 0
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                # Read and decode the image with TensorFlow
                img_raw = tf.io.read_file(file_path)
                img = tf.image.decode_image(img_raw)
            except tf.errors.InvalidArgumentError as e:
                os.remove(file_path)
                num_deleted += 1
    print(f"Total corrupted images removed (TensorFlow check): {num_deleted}")



def fn_data_augmentation():
    
    # Vertical & Horizontal Flips
    horizontal_flips = tf.keras.layers.RandomFlip('horizontal')
    vertical_flips = tf.keras.layers.RandomFlip('vertical')
    
    # Random Rotation
    rotation = tf.keras.layers.RandomRotation(0.1)  # Rotate by up to ±10%
    
    # Random Zoom
    zoom = tf.keras.layers.RandomZoom(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1))
    
    # Brightness Adjustment
    brightness = tf.keras.layers.RandomBrightness(factor=(-0.2, 0.2))  # Adjust brightness by ±20%
    
    # Contrast Adjustment
    contrast = tf.keras.layers.RandomContrast(factor=0.2)  # Adjust contrast by ±20%
    
    # Random Translation
    translation = tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1)

    #Assemble the augmentation steps
    augmentation_steps = [horizontal_flips, vertical_flips, rotation, zoom, brightness, contrast, translation]

    return augmentation_steps

#MAIN FUNCTION
if __name__ == '__main__':

    try:

        start_time = dt.now()
        print("Bone_Fracture_Model_Generator_Script Start Time : {}".format(start_time))

        ################################################################################
        ## Data Loading and Cleaning
        ################################################################################

        dataset_path = './datasets'
        model_path = './model_files'
        
        train_dir = os.path.join(dataset_path, 'train')
        validation_dir = os.path.join(dataset_path, 'val')
        test_dir = os.path.join(dataset_path, 'test')


        # Remove corrupted images from each dataset
        fn_remove_corrupted_images_tf(train_dir)
        fn_remove_corrupted_images_tf(validation_dir)
        fn_remove_corrupted_images_tf(test_dir)
        
        #define batch size, image size and image shape
        batch_size = 32
        img_size = (224, 224)
        img_shape = img_size + (3,)
        
        train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir, shuffle=True, batch_size=batch_size, image_size=img_size)
        validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir, shuffle=True, batch_size=batch_size, image_size=img_size)
        test_dataset = tf.keras.utils.image_dataset_from_directory(test_dir, shuffle=True, batch_size=batch_size, image_size=img_size)

        #Create the data augmentation steps
        data_augmentation = tf.keras.Sequential(fn_data_augmentation())

        # Optimize dataset performance with prefetching
        autotune = tf.data.AUTOTUNE
        train_dataset = train_dataset.prefetch(buffer_size=autotune)
        validation_dataset = validation_dataset.prefetch(buffer_size=autotune)
        test_dataset = test_dataset.prefetch(buffer_size=autotune)

        ################################################################################
        ## Data Model Setup & Training
        ################################################################################

        #Initialize the pre-trained model
        base_model = tf.keras.applications.MobileNetV3Large(input_shape=img_shape,
                                                            include_top=False,
                                                            weights='imagenet')

        # Freeze the first 100 layers of the base model
        base_model.trainable = True 
        for layer in base_model.layers[:100]: 
            layer.trainable = False

        #Setting up the inputs
        inputs = tf.keras.Input(shape=img_shape)
        x = data_augmentation(inputs)
        x = tf.keras.applications.mobilenet_v3.preprocess_input(x)
        x = base_model(x, training=False)

        #Adding a new pooling layer
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        x = global_average_layer(x)

        #Setting up and assigning the prediction layer
        prediction_layer = tf.keras.layers.Dense(1, activation='sigmoid')
        
        # Applying a Fully Connected Layer to predict the class
        outputs = prediction_layer(x)
        model = tf.keras.Model(inputs, outputs)

        #Model Compilation
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),  # Smaller learning rate
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5, name='accuracy')]
        )


        # Setting callbacks
        fine_tune_early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        fine_tune_model_checkpoint = ModelCheckpoint(filepath=os.path.join(model_path,'bone_fracture_model.keras'), monitor='val_loss', save_best_only=True)
        

        # Model Training and logging
        model_logs = model.fit(
            train_dataset,
            epochs=100,
            validation_data=validation_dataset,
            callbacks=[fine_tune_early_stopping, fine_tune_model_checkpoint],
            verbose=1
        )
        
        # Save fine-tuning history
        with open(os.path.join(model_path,'bone_fracture_model_logs.json'), 'w') as file:
            json.dump(model_logs.history, file)


        # Evaluate the fine-tuned model on the test dataset
        test_loss, test_accuracy = model.evaluate(test_dataset)

        # Gather ground truth and predictions from the test dataset
        ground_truth = np.concatenate([labels.numpy() for _, labels in test_dataset])
        predictions = np.concatenate([tf.where(model(X).numpy().flatten() < 0.5, 0, 1).numpy() for X, _ in test_dataset])

        # Print classification report
        print("Classification Report:")
        print(classification_report(ground_truth, predictions, target_names=class_names))


        print("Model Testing Completed  successfully.")
        
        end_time = dt.now()
        print("Bone_Fracture_Model_Generator_Script Completed:{}".format(end_time))
        duration = end_time - start_time
        td_mins = int(round(duration.total_seconds() / 60))
        print('The difference is approx. %s minutes' % td_mins)
     
    except Exception as e:
        
        error = "Bone_Fracture_Model_Generator_Script Failure :: Error Message {} with Traceback Details: {}".format(e,tb.format_exc())        
        print(error)        
        
        


