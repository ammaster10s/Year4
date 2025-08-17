import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import f1_score

class Helper:
    @staticmethod
    def preprocess_noise(image, method='median', kernel_size=3):
        if kernel_size % 2 == 0 or kernel_size < 1:
            raise ValueError("Kernel size must be an odd integer greater than 1.")

        if method == 'gaussian':
            # Apply Gaussian Blur
            denoised = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        elif method == 'median':
            # Apply Median Filter
            denoised = cv2.medianBlur(image, kernel_size)
        elif method == 'bilateral':
            # Apply Bilateral Filter
            denoised = cv2.bilateralFilter(image, kernel_size, 75, 75)
        else:
            raise ValueError("Unsupported denoising method. Use 'gaussian', 'median', or 'bilateral'.")

        return denoised

    @staticmethod
    def evaluate_segmentation_f1(model, eval_images, input_size):
        """
        Computes the F1-score for segmentation on given images.

        Args:
            model: Trained model for prediction.
            eval_images: List of tuples (image_path, ground_truth_mask).
            input_size: Tuple of (height, width) expected by the model.

        Returns:
            list: Per-image F1 scores for the model.
        """
        f1_scores = []

        for idx, (image_path, ground_truth_mask) in enumerate(eval_images):
            # Load and preprocess the image
            image = load_img(image_path, color_mode='grayscale', target_size=input_size)
            image_array = img_to_array(image) / 255.0

            # Extract 2D array and preprocess noise
            image_array_2d = image_array[:, :, 0]  # Remove channel dimension for OpenCV
            image_array_denoised = Helper.preprocess_noise(image_array_2d)  # Apply noise reduction

            # Restore channel and batch dimensions
            image_array = np.expand_dims(image_array_denoised, axis=-1)  # Add channel dimension
            image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

            # Predict mask
            predicted_mask = model.predict(image_array)[0, :, :, 0]

            # Resize ground truth mask to match predicted mask shape
            predicted_mask_shape = predicted_mask.shape  # (height, width)
            resized_ground_truth = np.array(
                Image.fromarray(ground_truth_mask).resize(predicted_mask_shape[::-1], Image.NEAREST)
            )

            # Flatten both masks for F1 score calculation
            ground_truth_flat = resized_ground_truth.flatten()
            predicted_flat = (predicted_mask > 0.5).astype(np.uint8).flatten()

            # Handle edge case where both are completely empty
            if ground_truth_flat.sum() == 0 and predicted_flat.sum() == 0:
                f1 = 1.0  # Perfect score for no regions
            else:
                f1 = f1_score(ground_truth_flat, predicted_flat, average='binary')

            f1_scores.append(f1)

        return f1_scores