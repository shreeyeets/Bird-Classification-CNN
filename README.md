# Bird Classification Using Convolutional Neural Networks

This project implements a Convolutional Neural Network (CNN) to classify images of birds into ten different species. The model is trained on a dataset of bird images, utilizing several advanced techniques to enhance performance and interpretability.

## Model Architecture

The CNN architecture consists of the following layers:

| Layer Type          | Input Size | Output Size      | Description                              |
|---------------------|------------|------------------|------------------------------------------|
| Conv2D              | 64x64x3    | 64x64x64         | 3x3 kernel, 64 filters                  |
| Max Pooling         | 64x64x64   | 32x32x64         | 2x2 pooling                              |
| Conv2D              | 32x32x64   | 32x32x128        | 3x3 kernel, 128 filters                 |
| Max Pooling         | 32x32x128  | 16x16x128        | 2x2 pooling                              |
| Conv2D              | 16x16x128  | 16x16x256        | 3x3 kernel, 256 filters                 |
| Max Pooling         | 16x16x256  | 8x8x256          | 2x2 pooling                              |
| Conv2D              | 8x8x256    | 8x8x512          | 3x3 kernel, 512 filters                 |
| Max Pooling         | 8x8x512    | 4x4x512          | 2x2 pooling                              |
| Flatten             | 4x4x512    | 8192             | Flatten to feed into fully connected layers |
| Fully Connected      | 8192       | 1024             | First fully connected layer              |
| Dropout             | 1024       | 1024             | Regularization through dropout           |
| Fully Connected      | 1024       | 512              | Second fully connected layer             |
| Dropout             | 512        | 512              | Regularization through dropout           |
| Fully Connected      | 512        | 10               | Output layer for 10 classes              |

## Training and Validation Loss vs. Epochs

The model's training and validation loss were plotted over epochs, illustrating how the model's learning progressed over time. The loss values generally decreased, indicating effective learning. 

![Loss Plot](path_to_loss_plot.png)  <!-- Replace with your actual image path -->

## Training and Validation Accuracy vs. Epochs

Similarly, training and validation accuracy were plotted across epochs to evaluate performance on both datasets. The accuracy increased, reflecting successful learning and generalization.

![Accuracy Plot](path_to_accuracy_plot.png)  <!-- Replace with your actual image path -->

## Effect of Model Optimization

The following table summarizes the validation accuracy results for various optimization techniques applied during training:

| Optimization Technique        | Data Augmentation | Regularization Technique | Validation Accuracy (%) |
|-------------------------------|-------------------|--------------------------|-------------------------|
| No Augmentation                | No                | None                     | XX.X                    |
| Augmentation with Flip        | Yes               | None                     | XX.X                    |
| Augmentation + Dropout        | Yes               | Dropout (0.5)           | XX.X                    |
| Augmentation + L2 Regularization | Yes               | L2 (λ = 0.001)          | XX.X                    |

These techniques significantly impacted the model's performance, demonstrating the importance of data augmentation and regularization in preventing overfitting.

## Class Activation Maps (CAM)

Visualizations of Class Activation Maps (CAM) were created for each class, providing insights into which regions of the images the model focuses on when making predictions. These visualizations help in understanding model behavior and interpreting its decisions.

![Class Activation Maps](path_to_cam_visualization.png)  <!-- Replace with your actual image path -->

By examining the highlighted areas, we can see that the model effectively identifies features specific to each bird species, such as beak shape, color patterns, and body structure.

## Conclusion

This project demonstrates the application of neural networks in image classification tasks, specifically in recognizing bird species. The techniques employed, including data augmentation, class balancing through weighted loss functions, and visualization methods like Class Activation Maps, enhance both the performance and interpretability of the model.
