import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def save_rendered_images(images_list, iteration, psnr_test, output_dir):
    """
    Save rendered images from a list of tensors.

    Args:
    - images_list (list of torch.Tensor): List of tensors containing images.
    - iteration (int): Iteration number for caption.
    - psnr_test (float): PSNR value for caption.
    - output_dir (str): Directory path where images will be saved.
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through images and save them individually
    for idx, image_tensor in enumerate(images_list):
        # Convert tensor to numpy array
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        
        # Create filename and caption
        caption = f"Iteration: {iteration}, PSNR: {psnr_test}"
        filename = os.path.join(output_dir, f"rendered_image_{idx}.png")

        # Save the image
        plt.imshow(image_np)
        plt.axis('off')  # Turn off axis numbers and ticks
        plt.title(caption)
        plt.savefig(filename, bbox_inches='tight')
        plt.close()  # Close the current plot to release memory
    return
