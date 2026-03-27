from src.dataset import LineOCRDataset

dataset = LineOCRDataset(images_dir="data/lines", labels_csv="data/lines/labels.csv")

# Get first item
img_tensor, label_tensor = dataset[0]

print(img_tensor.shape)   # e.g., [3, 28, variable_width]
print(label_tensor)       # e.g., [5, 12, 2, 8] -> encoded text indices