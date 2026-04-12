import numpy as np
from scipy.io import loadmat

def verify():
    boxes = loadmat('data/box_data.mat')['boxes']
    ops = loadmat('data/op_data.mat')['ops']
    
    num_samples = ops.shape[1]
    max_boxes = boxes.shape[1] // num_samples
    
    print(f"Total samples: {num_samples}")
    print(f"Max boxes per tree: {max_boxes}")
    
    engine_counts = []
    box_counts = []
    
    for i in range(num_samples):
        sample_boxes = boxes[:, i*max_boxes : (i+1)*max_boxes]
        # Valid boxes are those that aren't all zero (or at least have some non-zero feature)
        # Actually, let's check the ops to be sure.
        
        # Count boxes where the 12th feature (engine) is 1.0 (after scaling it becomes 1.0)
        # Wait, the scaling maps {0, 1} to {-1, 1}. So engine = 1.0
        engines = np.sum(sample_boxes[12, :] > 0.5)
        engine_counts.append(engines)
        
        # Count non-zero boxes
        count = 0
        for j in range(max_boxes):
            if np.any(sample_boxes[:, j] != 0):
                count += 1
        box_counts.append(count)

    print(f"Average engines per aircraft: {np.mean(engine_counts):.2f}")
    print(f"Min engines: {np.min(engine_counts)}")
    print(f"Max engines: {np.max(engine_counts)}")
    print(f"Percentage with 0 engines: {np.sum(np.array(engine_counts) == 0) / num_samples * 100:.2f}%")
    print(f"Average box count: {np.mean(box_counts):.2f}")

if __name__ == "__main__":
    verify()
