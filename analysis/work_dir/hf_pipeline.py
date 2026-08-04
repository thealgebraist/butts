import cv2
import numpy as np
import sys
import gc
from transformers import pipeline
import torch

def process_hf_video(input_video, output_video):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error opening video")
        sys.exit(1)
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0
    
    # Target resolution for fast inference
    W_target, H_target = 256, 512
    W_orig, H_orig = 512, 1024
    
    print("Loading Hugging Face DETR Panoptic Segmentation model...")
    # Use CPU for now since MPS might not be fully supported for all operations in DETR
    segmenter = pipeline("image-segmentation", model="facebook/detr-resnet-50-panoptic", device=-1)
    print("Model loaded successfully.")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (W_orig, H_orig))
    
    count = 0
    print("Starting Deep Learning frame segmentation...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Downsample for faster inference
        small_frame = cv2.resize(frame, (W_target, H_target))
        # Convert BGR to RGB for Hugging Face
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        from PIL import Image
        pil_img = Image.fromarray(rgb_frame)
        
        # Run inference
        results = segmenter(pil_img)
        
        # Find the mask belonging to the soda can (labeled as 'bottle' or 'cup')
        can_mask = None
        for res in results:
            if res['label'] in ['bottle', 'cup', 'can', 'jar', 'vase']:
                can_mask = np.array(res['mask'])
                break
                
        if can_mask is None:
            # If no can detected, output black mask
            can_mask = np.zeros((H_target, W_target), dtype=np.uint8)
        else:
            can_mask = (can_mask > 0).astype(np.uint8) * 255
            
        # Upscale the mask back to HD resolution
        hd_mask = cv2.resize(can_mask, (W_orig, H_orig), interpolation=cv2.INTER_NEAREST)
        
        # Convert to a continuous 3-channel alpha float multiplier
        alpha = hd_mask.astype(np.float32) / 255.0
        alpha_3ch = np.stack([alpha, alpha, alpha], axis=-1)
        
        # Apply the mask natively to the original HD color video
        masked_frame = (frame.astype(np.float32) * alpha_3ch).astype(np.uint8)
        
        out.write(masked_frame)
        
        count += 1
        if count % 100 == 0:
            print(f"Segmented {count} frames via Hugging Face...")
            
    cap.release()
    out.release()
    print("Deep Learning Panoptic Segmentation complete.")

if __name__ == "__main__":
    process_hf_video(sys.argv[1], sys.argv[2])
