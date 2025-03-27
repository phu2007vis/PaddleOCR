

def gether_list_polys(results):
    return [p['points'] for p in results]


import numpy as np
import paddle

def process_in_batches(images_preprocessed, 
                       shape_list_preprocessed,
                       model, 
                       post_process_class,
                       merger,
                       locations,
                       batch_size=40):
    """
    Process images in smaller batches for inference and merging.
    
    Args:
        images_preprocessed (list): List of preprocessed images.
        shape_list_preprocessed (list): List of shape info for each image.
        model: Paddle model for inference.
        post_process_class: Post-processing class/function for predictions.
        merger: Instance of Merger class to merge polygons.
        locations: List of slice locations corresponding to polygons.
        batch_size (int): Number of images to process per batch (default: 4).
    
    Returns:
        list: Final merged polygons.
    """
    num_images = len(images_preprocessed)
    all_post_result = []
    all_phuoc_post_result = []
    # Process images in batches
    
    for start_idx in range(0, num_images, batch_size):
     
        end_idx = min(start_idx + batch_size, num_images)
        batch_images = images_preprocessed[start_idx:end_idx]
        batch_shapes = shape_list_preprocessed[start_idx:end_idx]
        

        # Stack into NumPy arrays for the current batch
        images = np.stack(batch_images, axis=0)  # Shape: (batch_size, H, W, C)
        shape_list = np.stack(batch_shapes, axis=0)  # Shape: (batch_size, ...)

        # Convert to Paddle tensor
        images = paddle.to_tensor(images)

        # Run inference
        preds = model(images)

        # Post-process predictions
        post_result = post_process_class(preds, shape_list)
        all_post_result.extend(post_result)
        all_phuoc_post_result.extend([p['points'] for p in post_result])

        # Merge polygons for this batch
    
    final_box = merger(polygons=all_phuoc_post_result, slice_locations=locations)

    return final_box,post_result
