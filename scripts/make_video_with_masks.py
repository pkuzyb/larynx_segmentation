
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
import cv2

import numpy as np
import random
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
import matplotlib.pyplot as plt
from tqdm import tqdm 
import gc
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def read_video_frames(video_path, resized_width=502, resized_height=502):
    video = cv2.VideoCapture(video_path)
    nframe = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frames = []
    for i in tqdm(range(nframe), desc="Reading frames", unit="frame"):
        ret, frame = video.read()
        if not ret:
            print(f"Failed to read frame {i}")
            break  
        frame_resized = cv2.resize(frame, (resized_height, resized_width), interpolation=cv2.INTER_LINEAR) 
        frames.append(frame_resized)
    video.release() 
    return frames, fps 


def predict_frames_masks(frames, device, weights_path, batch_size=1, out_path='./predictions3.pt'):
    model = prepare_model(weights_path=weights_path, num_classes=7, device=device)
    model.to(device)
    model.eval() 
    predictions = []

    with torch.no_grad():
        for i in tqdm(range(0, len(frames), batch_size), desc="Processing Batches"):
            batch_frames = frames[i:i + batch_size]
            batch_predictions = []

            for frame in batch_frames:
                tensor_frame = torch.as_tensor(frame, dtype=torch.float32).permute(2, 0, 1)
                tensor_frame = tensor_frame.unsqueeze(0).to(device)
                outputs = get_predictions(model, tensor_frame)
                batch_predictions.append(outputs)

            predictions.extend(batch_predictions)
            del batch_predictions  # Free up memory
            gc.collect()  # Garbage collection to free up memory
            torch.cuda.empty_cache()  # Clear cache if using GPU

    # Save predictions to a .pt file
    torch.save(predictions, out_path)
    return predictions

def overlay_masks_on_frames(frames, predictions, score_threshold=0.5, mask_threshold=0.2, output_file='overlayed_frames2.npy'):
    overlayed_frames = []
    
    for image, prediction in zip(frames, predictions):
        image_with_mask = image.copy()
        masks = prediction[0]['masks']
        scores = prediction[0]['scores']
        labels = prediction[0]['labels']

        for i in range(len(labels)):
            mask = masks[i, 0].detach().cpu().numpy()
            score = scores[i].detach().cpu().numpy()
            label = labels[i].item()  
            color = color_mapping.get(label, (255, 255, 255))
            
            if score > score_threshold:
                color_scaled = np.array(color, dtype=np.uint8)
                image_with_mask[:, :, 0][mask > mask_threshold] = color_scaled[0]
                image_with_mask[:, :, 1][mask > mask_threshold] = color_scaled[1]
                image_with_mask[:, :, 2][mask > mask_threshold] = color_scaled[2]
        
        output_image = np.hstack([image, image_with_mask])
        overlayed_frames.append(output_image)
    
    # Convert the list of overlayed frames to a NumPy array
    overlayed_frames_np = np.array(overlayed_frames)
    
    # Save the overlayed frames as a .npy file
    np.save(output_file, overlayed_frames_np)
    print(f"Overlayed frames saved to {output_file}")

    return overlayed_frames



def write_video_with_masks(output_path, frames, fps):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()
    print('The video saved.')


# def process_video_with_masks(input_video_path, output_video_path, device):
#     frames, fps = read_video_frames(input_video_path)
#     masks = predict_frames_masks(frames, device)
#     overlayed_frames = overlay_masks_on_frames(frames, masks)
#     write_video_with_masks(output_video_path, overlayed_frames, fps)


def longest_path_on_contour(contour, start_point, end_point, output_path='./path_on_contour'):
    """
    Finds the longest path along the contour between two specified points
    by calculating distances in both directions.

    Parameters:
        contour (numpy.ndarray): A single contour obtained from cv2.findContours, with shape (num_points, 2).
        start_point (tuple): Starting point of the path on the contour.
        end_point (tuple): Ending point of the path on the contour.
        output_path (str): Path to save the output image.

    Returns:
        tuple: List of points forming the longest path and the path length.
    """
    # Ensure the start and end points are tuples for comparison
    start_point = tuple(start_point)
    end_point = tuple(end_point)

    # Find the indices of the start and end points
    start_index = np.where((contour == start_point).all(axis=1))[0][0]
    end_index = np.where((contour == end_point).all(axis=1))[0][0]

    # Calculate distance from start to end along the contour
    if start_index < end_index:
        path1 = contour[start_index:end_index + 1]
    else:
        path1 = np.concatenate((contour[start_index:], contour[:end_index + 1]))

    distance1 = np.sum(np.sqrt(np.sum(np.diff(path1, axis=0) ** 2, axis=1)))

    # Calculate distance from end to start along the contour
    if end_index < start_index:
        path2 = contour[end_index:start_index + 1]
    else:
        path2 = np.concatenate((contour[end_index:], contour[:start_index + 1]))

    # Reorder path2 to go from start to end point
    if end_index < start_index:
        path2 = np.flipud(path2)

    distance2 = np.sum(np.sqrt(np.sum(np.diff(path2, axis=0) ** 2, axis=1)))

    # Determine which path is longer
    if distance1 >= distance2:
        longest_path = path1
        longest_length = distance1
    else:
        longest_path = path2
        longest_length = distance2

    # Plotting the contour and the longest path
    plt.figure(figsize=(8, 8))
    plt.plot(contour[:, 0], -contour[:, 1], color='green', linewidth=2, label='Contour')
    
    longest_path_array = np.array(longest_path)
    plt.plot(longest_path_array[:, 0], -longest_path_array[:, 1], color='red', linewidth=2, label='Longest Path')
    
    plt.scatter(*start_point, color='blue', label='Start Point')
    plt.scatter(*end_point, color='orange', label='End Point')

    plt.legend()
    plt.title('Longest Path on Contour')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.axis('equal')
    plt.grid()

    # Save the resulting plot
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    return longest_path.tolist(), longest_length


def find_extreme_points(contour):
    """
    Find the two extreme points along the x-axis and y-axis in a contour.

    Parameters:
        contour (array): A single contour, as returned by cv2.findContours.

    Returns:
        dict: A dictionary containing the extreme points: leftmost, rightmost, topmost, and bottommost.
    """
    leftmost = tuple(contour[np.argmin(contour[:, 0])])   # Point with minimum x-coordinate
    rightmost = tuple(contour[np.argmax(contour[:, 0])])  # Point with maximum x-coordinate
    topmost = tuple(contour[np.argmin(contour[:, 1])])    # Point with minimum y-coordinate
    bottommost = tuple(contour[np.argmax(contour[:, 1])]) # Point with maximum y-coordinate


    return leftmost, rightmost, topmost, bottommost

def smooth_contour(contour):
    resX, resY = regularize_Bsplines(np.array(contour), 3)
    return np.array([resX, resY]).T



def calculate_external_contour(binary_mask, output_path):
    """
    Visualizes the external contour of a binary mask and the mask itself using Matplotlib.

    Parameters:
    binary_mask (numpy.ndarray): The binary mask with foreground (1) and background (0).
    output_path (str): Path to save the visualized output with contour overlay.
    """
    # Convert binary mask from 1s and 0s to 255 and 0 for proper contour detection
    mask_for_contour = (binary_mask * 255).astype(np.uint8)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask_for_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Create a figure for plotting
    plt.figure(figsize=(10, 10))

    # Plot the binary mask
    plt.subplot(1, 2, 1)
    plt.imshow(binary_mask, cmap='gray')
    plt.title("Binary Mask")
    plt.axis('off')

    # Draw the contours if they exist
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)  # Find the largest contour by area
        contour_points = np.squeeze(largest_contour[:, 0, :]) # Extract x, y coordinates
        contour_points = np.vstack([contour_points, contour_points[0]])


        # Plot the largest contour
        plt.subplot(1, 2, 2)
        plt.imshow(binary_mask, cmap='gray')  # Plot mask again to overlay the contour
        plt.plot(contour_points[:, 0], contour_points[:, 1], color='red', linewidth=2, label='Largest Contour')
        plt.title("Largest Contour on Mask")
        plt.axis('off')
        plt.legend()

    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()  # Close the plot to free memory
    print(f"Visualization saved as {output_path}")
    return(contour_points)


def visualize_contour_on_mask(mask, contour, output_path):
    """
    Visualizes the largest contour on top of the binary mask.

    Parameters:
    mask (numpy.ndarray): The binary mask on which to draw the contour.
    contour (numpy.ndarray): The contour to be drawn.
    output_path (str): Path to save the visualized output.
    """
    # Create a 3-channel image from the binary mask
    contour_image = np.zeros((*mask.shape, 3), dtype=np.uint8)  # Create an RGB image
    contour_image[mask > 0] = [255, 255, 255]  # Set mask pixels to white

    if contour is not None:  # Check if a contour was found
        # Draw the largest contour in blue
        cv2.drawContours(contour_image, [contour], contourIdx=-1, color=(255, 0, 0), thickness=2)  # Blue color for contour

    # Save the output
    cv2.imwrite(output_path, contour_image)
    print(f"Contour visualized and saved as {output_path}")

def calculate_contours(predictions,score_threshold =0.5):
    """
    Calculate contours based on the Mask R-CNN predictions.

    Parameters:
    predictions (list): Output from the Mask R-CNN model.

    Returns:
    dict: A dictionary containing contours for specified labels.
    """
    contour_dict = {}

    # Get masks and labels
    masks = predictions[0]['masks'].detach().cpu().numpy()
    labels = predictions[0]['labels'].detach().cpu().numpy()
    scores = predictions[0]['scores'].detach().cpu().numpy()

    valid_indices = np.where(scores > score_threshold)[0]
    labels = labels[valid_indices]
    masks = masks[valid_indices]

    binary_masks = (masks > 0.1).astype(np.uint8)

    # Get contours for thyroid cartilage and ventricle (labels 1 and 2)
    for label in [1, 2]:
        label_indices = np.where(labels == label)[0]  # Find indices of the specified label
        if len(label_indices) > 0:
            for idx in label_indices:
                mask = binary_masks[idx, 0] # Use the binary mask
                largest_contour=calculate_external_contour(mask,str(idx)+'mask.png')
                contour_dict['thyroid cartilage' if label == 1 else 'ventricle'] = largest_contour

    # Process for label 3 (inferior vocal folds)
    label_indices = np.where(labels == 3)[0]
    if len(label_indices) > 0:
        for idx in label_indices:
            mask = binary_masks[idx, 0]  # Use the binary mask
            largest_contour=calculate_external_contour(mask,str(idx)+'mask.png')
            leftmost, rightmost, _, _ = find_extreme_points(largest_contour)
            path, _ = longest_path_on_contour(largest_contour, leftmost, rightmost)
            contour_dict['inferior vocal folds']=  path

    # Process for label 4 (arytenoid cartilage)
    label_indices = np.where(labels == 4)[0]
    if len(label_indices) > 0:
        for idx in label_indices:
            mask = binary_masks[idx, 0]  # Use the binary mask
            largest_contour=calculate_external_contour(mask,str(idx)+'mask.png')
            _, _, topmost, bottommost = find_extreme_points(largest_contour)
            path, _ = longest_path_on_contour(largest_contour, bottommost, topmost)
            contour_dict['arytenoid cartilage'] = path

    # Process for labels 5 and 6 (epiglottis)
    for label in [5, 6]:
        label_indices = np.where(labels == label)[0]
        if len(label_indices) > 0:
            for idx in label_indices:
                mask = binary_masks[idx, 0]  # Use the binary mask
                largest_contour=calculate_external_contour(mask,str(idx)+'mask.png')

                min_y_point, max_y_point, _, _ = find_extreme_points(mask)
                if min_y_point is not None and max_y_point is not None:
                    if label == 5:
                        _, _, bottommost, topmost = find_extreme_points(largest_contour)
                        path, _ = longest_path_on_contour(largest_contour, bottommost, topmost)
                    elif label == 6:
                        leftmost, rightmost, _, _ = find_extreme_points(largest_contour)
                        path, _ = longest_path_on_contour(largest_contour, rightmost, leftmost)
                    if 'epiglottis' not in contour_dict:
                        contour_dict['epiglottis'] = []
                    contour_dict['epiglottis'].extend(path)
    contour_dict['epiglottis']=np.array(contour_dict['epiglottis']) 

    return contour_dict

def plot_contours_on_image(image_path, contour_dict, output_path):
    """
    Plots contours from the contour dictionary on the input image and saves the plot.

    Parameters:
    image_path (str): Path to the input image.
    contour_dict (dict): Dictionary containing contour coordinates.
    output_path (str): Path to save the plotted image.
    """
    # Load the original image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for Matplotlib

    # Define a color mapping for each contour type
    color_mapping = {
        'thyroid cartilage': (1, 0, 0),  # Red
        'ventricle': (0, 1, 0),          # Green
        'vocal folds': (0, 0, 1), # Blue
        'arytenoid cartilage': (0, 1, 1), # Cyan
        'epiglottis': (1, 0, 1)        # Magenta
    }

    # Create a new figure
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    for key, contour in contour_dict.items():
        #contour = smooth_contour(contour)
        contour =np.array (contour)
        color = color_mapping.get(key, (1, 1, 1))  # Default to white if the key is not found
        plt.plot(contour[:, 0],contour[:, 1], color=color, linewidth=1, label=key,marker='o', markersize=1)  # Plot each contour

    plt.axis('off')  # Hide the axes
    plt.legend()     # Show legend
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Contours plotted and saved as {output_path}")

# Function to load and prepare the image
def load_image(image_path, device):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1)
    # For COCO pretrained weights
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = (image / 255 - torch.tensor(mean).view(-1, 1, 1)) / torch.tensor(std).view(-1, 1, 1)
    return [image.to(device)]

def load_image_unnorm(image_path, device):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1)
    return [image.to(device)]

# Function to prepare the Mask R-CNN model
def prepare_model(weights_path=None, num_classes=7, device='cuda'):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256  # Default value, can be adjusted
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    # Load custom weights if provided
    if weights_path:
        print(weights_path)
        model.load_state_dict(torch.load(weights_path, map_location=device,weights_only=True))
    
    model.to(device)
    model.eval()
    return model

# Function to run the model and get predictions
def get_predictions(model, images):
    with torch.no_grad():
        predictions = model(images)
    return predictions

color_mapping = {
    1: (255, 0, 0),    
    2: (0, 255, 0),    
    3: (0, 0, 255),     
    4: (0, 255, 255),   
    5: (255, 0, 255),   
    6: (255, 255, 0)
}

def visualize_mask(image, predictions, output_path, score_threshold=0.5, mask_threshold=0.2):
    
    image_np = image[0].swapaxes(0, 2).swapaxes(0, 1).detach().cpu().numpy().astype(np.uint8)
    output_image = image_np.copy()

    masks = predictions[0]['masks']
    scores = predictions[0]['scores']
    labels = predictions[0]['labels']

    for i in range(len(labels)):
        mask = masks[i, 0].detach().cpu().numpy()
        score = scores[i].detach().cpu().numpy()
        label = labels[i].item()  # Assuming labels are indices or label names

        # Set the color, default to white if not found
        color = color_mapping.get(label, (255, 255, 255))

        if score > score_threshold:
            # Convert color values directly to uint8
            color_scaled = np.array(color, dtype=np.uint8)
            output_image[:, :, 0][mask > mask_threshold] = color_scaled[0]
            output_image[:, :, 1][mask > mask_threshold] = color_scaled[1]
            output_image[:, :, 2][mask > mask_threshold] = color_scaled[2]

    # Display and save output
    # cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Output', 1024, 1024)
    # cv2.imshow('Output', cv2.resize(np.hstack([image_np, output_image]), (1024*2, 1024)))
    # cv2.waitKey()
    cv2.imwrite(output_path, output_image)
    # cv2.destroyAllWindows()


def plot_and_save_binary_mask(mask, filename="binary_mask_plot.png"):
    """
    Plots and saves a binary mask using Matplotlib.

    Parameters:
    mask (np.array): A binary mask (2D numpy array).
    filename (str): The name of the file to save the plot as (default: "binary_mask_plot.png").
    """
    plt.imshow(mask, cmap='gray')
    plt.title("Binary Mask")
    plt.axis("off")  # Turn off axis labels for clarity
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Mask saved as {filename}")


if __name__ == "__main__":
    weights_path = './maskrcnn_WeiLoss22_batch1_mskthr0.5/weights_epoch_6.pth'  # Path to custom model weights
#  /Users/yubin/Library/CloudStorage/GoogleDrive-yubinzha@usc.edu/My Drive/dissertation/dissertation_data/lary_seg/laryseg_annotation_second_attempt/maskrcnn_larynx_segmnetation/video_frame_viewer2.py
    ## single image prediction
    image_path = '../data/test/yrb_02_006_00431.jpg'  # Path to your input image
    output_path = './yrb_02_006_00431_val3.png'  # Where to save the result
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    images = load_image(image_path, device)


    model = prepare_model(weights_path=weights_path, num_classes=7, device=device)
    outputs = get_predictions(model, images)
    print(outputs[0]['scores'])
    print(outputs[0]['labels'])
    images_unnorm = load_image_unnorm(image_path, device)

    visualize_mask(images_unnorm, outputs, output_path, score_threshold=0.5, mask_threshold = 0.1)
    

    # contour_dict = calculate_contours(outputs)
    # plot_contours_on_image(image_path, contour_dict, output_path)

    # mask=outputs[0]['masks'][5, 0].detach().cpu().numpy()
    # mask=np.squeeze(np.where(mask > 0.6, 1, 0).astype(np.uint8))
    # visualize_external_contour(mask,'./5_mask.png')
    # plot_and_save_binary_mask(mask, filename="my_binary_mask.png")
    # min_y_point, max_y_point,_,_ = find_extreme_points(mask)
    # print(min_y_point, max_y_point)
    # path=shortest_path_on_mask(mask, max_y_point, min_y_point, save_path="shortest_path.png")



    # visualize_mask(images, outputs, './masks2_upscaled_val2.png', score_threshold=0.5, mask_threshold = 0.1)
    # video_path = './yrb_05_19.avi'
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # frames, fps = read_video_frames(video_path)
    # predictions = predict_frames_masks(frames, device,weights_path,out_path='./yrb_05_19.pt')
    # overlayed_frames = overlay_masks_on_frames(frames, predictions)
    # write_video_with_masks('./yrb_05_19_masks.avi', overlayed_frames, fps)