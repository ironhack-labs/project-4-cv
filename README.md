# Parking Lot Detection

## Objective of the Project
The goal of this project is to build a computer vision model that detects and predicts empty or occupied parking spaces, aimed at improving intelligent parking management systems.

## Dataset Overview
The dataset consists of images and a video of a parking lot, labeled as empty or not empty.
- The images are organized to indicate the occupancy status.
- The video captures real-time changes in parking space occupancy.
- Some images include cropped regions (masks) that focus on specific parking spaces for accurate detection.

## Challenges & Solutions

### Challenge 1: Defining the Computer Vision Workflow
- **Challenge**: While the business goal was clear as a classification problem, determining the most effective computer vision workflow posed a challenge.
- **Solution**: Transfer learning was selected as the optimal approach, utilizing pre-trained models like MobileNet for faster development and improved performance.

### Challenge 2: Execution Time
- **Challenge**: The large dataset led to significant processing times, slowing down the overall progress.
- **Solution**: Optimized the workflow by limiting the dataset size and refining the model training process.

### Challenge 3: Google Colab Limitations
- **Challenge**: Google Colab's environment faced limitations when handling video and multi-frame loading, which affected performance.
- **Solution**: Reduced the number of frames processed after video loading, significantly improving efficiency.


## Methodology

### **ETL (Extract, Transform, Load)**
- Implemented functions to load the dataset from Google Drive into the Colab environment.
- Data is organized into two folders: "empty" (for empty parking spots) and "not empty" (for occupied spots).

### **Modeling**

#### **Train-Test Split**
- Class names were automatically derived from the folder names using the `class_indices` attribute.
- The ImageDataGenerator class was used to split the dataset into training and validation sets.
- Batch size and target image size were specified, and pixel values were normalized to the range [0, 1].

#### **Model Selection**

1. **First Model: CNN**
   - A Convolutional Neural Network (CNN) was designed with sequential layers:
     - **Input Layer**: Takes images with dimensions.
     - **Convolutional Layer (Conv2D)**: Extracts basic features like edges, lines, and textures.
     - **ReLU Activation Layer**: Introduces non-linearity by setting negative values to zero.
     - **MaxPooling2D Layer**: Reduces spatial dimensions and enhances feature extraction.
     - **Flatten Layer**: Converts the 2D image matrix into a 1D vector for the fully connected layers.
     - **Dense Layer**: Learns complex patterns and relationships in the data.
     - **Sigmoid Output Layer**: Produces a binary output (0 for empty, 1 for not empty).

2. **Second Model: MobileNet (Transfer Learning)**
   - Utilized a pre-trained MobileNet model (without top layers) for feature extraction.
   - Added a global average pooling layer and a dense layer with ReLU activation, followed by a final dense layer with a sigmoid activation function for binary classification.
   - The model was compiled with the Adam optimizer and binary cross-entropy loss and trained for 10 epochs.

### **Model Evaluation**

1. **Confusion Matrix**  
   - The model shows a balanced distribution between both classes, with true positives slightly exceeding other cases. 
   - Focus is on reducing false negatives while improving accuracy, precision, and recall.

2. **Classification Report**  
   - The model's performance shows difficulty distinguishing between the two classes, with F1-scores of 0.59 for the empty class and 0.51 for the not empty class. 
   - **Next Steps**: Perform data augmentation and fine-tune the model to improve classification accuracy.

3. **Predictions on Video**
   - MobileNet model used to predict parking space occupancy in a video, with empty spots highlighted in green and occupied spots marked in red.
   - **Improvements**: Test the model on a larger dataset to evaluate its performance at scale.

4. **Predicted Outcomes on Images**
   - All images were predicted correctly by the model.

5. **Next Steps**
   - Explore alternative transfer learning models to further enhance classification performance.
   - Implement fine-tuning on model weights specific to parking space detection.
   - Apply data augmentation techniques to improve generalization.
   - Use additional evaluation metrics, such as the ROC curve, for a more comprehensive performance analysis.
   - Focus on enhancing key metrics in the classification report, including precision, recall, and accuracy.

## Deliverables

1. **Python Code**  
   Conducted the analysis and data modeling using Python and Jupyter Notebook. The code includes data preprocessing, model training, and evaluation, showcasing the model's performance.

2. **Presentation**  
   A detailed PowerPoint presentation summarizing the findings and insights. It includes data visualizations, the confusion matrix, model performance metrics, and predicted outcomes with visual indicators (green and red boxes) on video frames.

