import torch
import cv2
import numpy as np
from engine import LoFTR

def load_image(image_path: str) -> torch.Tensor:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (256, 256)) 
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    return image_tensor

def draw_matches(ref_image, query_image, keypoints0, keypoints1, matches):
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_GRAY2BGR)
    query_image = cv2.cvtColor(query_image, cv2.COLOR_GRAY2BGR)
    for i, match in enumerate(matches):
        pt1 = (int(keypoints0[i][0]), int(keypoints0[i][1]))
        pt2 = (int(keypoints1[match][0]), int(keypoints1[match][1]))
        cv2.line(ref_image, pt1, pt2, (0, 255, 0), 1)
    return ref_image, query_image

def infer(model: LoFTR, ref_image_path: str, query_image_path: str):
    ref_image = load_image(ref_image_path)
    query_image = load_image(query_image_path)

    model.eval()
    with torch.no_grad():
        outputs = model({'image0': ref_image, 'image1': query_image})
    
    scores = outputs['scores']
    feat0 = outputs['feat0']
    feat1 = outputs['feat1']

    best_match = torch.argmax(scores, dim=-1)
    
    print("Best match indices:", best_match)

    ref_image_np = ref_image.squeeze().numpy() * 255.0
    query_image_np = query_image.squeeze().numpy() * 255.0
    keypoints0 = feat0.squeeze().permute(1, 2, 0).cpu().numpy()
    keypoints1 = feat1.squeeze().permute(1, 2, 0).cpu().numpy()

    ref_image_with_matches, query_image_with_matches = draw_matches(
        ref_image_np, query_image_np, keypoints0, keypoints1, best_match.cpu().numpy()
    )

    cv2.imshow("Reference Image with Matches", ref_image_with_matches)
    cv2.imshow("Query Image with Matches", query_image_with_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    
    model = LoFTR()
    
    # Load model weights
    # model.load_state_dict(torch.load('path_to_model_weights.pth'))

    ref_image_path = r"C:\Users\ASUS\OneDrive\Desktop\QATM\dataset_root\Google_17_sample.jpg"
    query_image_path = r"C:\Users\ASUS\OneDrive\Desktop\QATM\dataset_root\query_image.jpg"

    infer(model, ref_image_path, query_image_path)