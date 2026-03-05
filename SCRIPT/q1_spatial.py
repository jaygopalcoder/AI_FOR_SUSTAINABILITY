import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import geopandas as gpd
import rasterio

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(BASE_DIR,"archive","delhi_ncr_region.geojson")

# original delhi_ncr_region.geojson file stored
gdf = gpd.read_file(path)

# QUESTION 1


# changing CRS to epsg = 32644 as it mentioned and for 2d projection(in metres for 60x60 km grid) of spatial data and for plotting
gdf_m = gdf.to_crs(epsg=32644)


# now we are going to plot this data 
# for this we define a canvas "ax" where we plot the data with overlay of grid lines
fig,ax = plt.subplots(figsize=(10,10))
gdf_m.plot(ax = ax, cmap="tab20", edgecolor="black")


# now for grid lines we first find the boundary of our spatial flatten(epsg=32644) data
minx, miny, maxx, maxy = gdf_m.total_bounds # Return a tuple containing minx, miny, maxx, maxy values for the bounds of the series as a whole

# vertical dotted grid line
for x in np.arange(minx, maxx, 60000):
    ax.axvline(x, color = "red", linestyle = "--", linewidth = 0.5)

# horizontal dotted grid line
for y in np.arange(miny, maxy, 60000):
    ax.axhline(y, color = "red", linestyle = "--", linewidth = 0.5)

plt.title("Delhi NCR Region(60 X 60 km Grid)", fontsize=12, color='black', loc='center')
plt.xlabel("EAST ->",fontsize = 12)
plt.ylabel("NORTH ->",fontsize = 12)
plt.show() 

 # QUESTION 1.2 
#●	Filter satellite images whose center coordinates fall inside the region. (1 mark)
# for this first we have to extract image co-ordinates(lat,lon) from their name.

from shapely.geometry import Point

rgb_dir = os.path.join(BASE_DIR, "archive", "rgb")
image_data = []
for filename in os.listdir(rgb_dir): 

    if filename.endswith(".png"):
        parts = filename.replace(".png","").split("_")
        lat,lon = float(parts[0]),float(parts[1])
        point = Point(lon,lat)
        image_data.append({"filename":filename, "geometry":point})

gdf_images = gpd.GeoDataFrame(image_data,crs="EPSG:4326")
print(f"TOTAL NO OF IMAGES: {len(gdf_images)}")


# combining whole geometry co-ordinates of each individual image into single dataset
combined_ncr_boundary = gdf.geometry.union_all()
gdf_filtered = gdf_images[gdf_images.geometry.within(combined_ncr_boundary)] # filtering image inside the boundary
print(f"TOTAL NO OF IMAGES INSIDE THE DELFHI NCR BOUNDARY: {len(gdf_filtered)}")
print(f"IOTAL NO OF IMAGES OUTSIDE THE DELHI NCR BOUNDARY: {len(gdf_images)-len(gdf_filtered)}")


# QUESTION 2
#Q 2.1 ●	For each image, extract the  128×128 corresponding land-cover patch from land_cover.tif using its center coordinate (2 marks)
tif_path = os.path.join(BASE_DIR, "archive", "worldcover_bbox_delhi_ncr_2021.tif")
dataset = rasterio.open(tif_path)
affine_matrix = dataset.transform
x,y = affine_matrix[2], affine_matrix[5]
row,column = dataset.index(x,y)
print(row,column)


from rasterio.windows import Window

# function for extracting 128 X 128 land cover patch
def extract_patch(dataset, lon, lat, patch_size=128):
    
    # because the patch we will cut is in pixels hence we need to covert degrees into numpy array row and col corresponding to that point using affine transformation 
    # lon/lat are input in degrees (EPSG:4326)
    row_idx, col_idx = dataset.index(lon, lat)
    
    # Calculate the top-left corner of the 128x128 window because col_idx and row_idx is centre co-ordinate so for top left corner we need to move half of the patch size to upper and half of patch size to the left. 
    
    col_off = col_idx - (patch_size // 2)
    row_off = row_idx - (patch_size // 2)
    
    # 3. Create the Window objectn- start from top-left corner(col_off,row_off) move to 128(or custom patch size)unit right move to 128(or custom patch size ) down
   
    win = Window(col_off, row_off, patch_size, patch_size)
    
    # 4. Read the data from 'Band 1' within this window
    # boundless=True prevents errors if the window goes slightly outside the TIF edge
    patch = dataset.read(1, window=win, boundless=True, fill_value=0)
    
    return patch

from scipy import stats
from tqdm import tqdm # This adds a progress bar for seeing scaning speed

def get_dominant_label(src, lon, lat):
    # Reuse the extract_patch function we wrote
    patch = extract_patch(src, lon, lat)
    
    # mode calculation (axis=None flattens the 128x128 to a long list)
    mode_res = stats.mode(patch, axis=None, keepdims=True)
    return int(mode_res.mode.item())

# applying this to filtered DataFrame
tqdm.pandas() # Initialize progress bar for pandas
gdf_filtered['raw_label'] = gdf_filtered.progress_apply(lambda row: get_dominant_label(dataset, row.geometry.x, row.geometry.y), axis=1)

label_map = {
    10: "Vegetation", 20: "Vegetation", 30: "Vegetation",
    40: "Cropland",
    50: "Built-up",
    60: "Others", 70: "Others", 90: "Others", 95: "Others", 100: "Others",
    80: "Water"
}

gdf_filtered['label_name'] = gdf_filtered['raw_label'].map(label_map)

from sklearn.model_selection import train_test_split

#spliting the dataframe itself
train_df, test_df = train_test_split(
    gdf_filtered, 
    test_size=0.40,      # 40% for testing
    random_state=42,    # it is used so that every time same set of data is considered for training and testing you can choose any number because ml model choose data randomly
    stratify=gdf_filtered['label_name'] # it makes sure that total data stored in every label is choosen 40% for testing and 60% for train 
)

print(f"Training samples: {len(train_df)}")
print(f"Testing samples: {len(test_df)}")

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
sns.countplot(data=gdf_filtered, x='label_name', order=['Built-up', 'Vegetation', 'Water', 'Cropland', 'Others'])
plt.title("Land Use Class Distribution (Delhi-NCR)")
plt.ylabel("Number of Images")
plt.show()

# Save for the CNN training phase
dataset_csv_path = os.path.join(BASE_DIR, "land_use_dataset.csv")
gdf_filtered.to_csv(dataset_csv_path , index=False)

# Q3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import os
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RGB_DIR = rgb_dir
# Map names to numbers for the model
CLASS_MAP = {"Built-up": 0, "Vegetation": 1, "Water": 2, "Cropland": 3, "Others": 4}
INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}

# 2. Dataset Class
class LandUseDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.df = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        image = Image.open(img_path).convert("RGB")
        label = CLASS_MAP[row['label_name']]
        
        if self.transform:
            image = self.transform(image)
        return image, label

# 3. Augmentation & Normalization (Standard for ResNet)
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 4. Prepare DataLoaders
train_dataset = LandUseDataset(train_df, RGB_DIR, transform=data_transforms['train'])
test_dataset = LandUseDataset(test_df, RGB_DIR, transform=data_transforms['test'])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 5. Model: ResNet-18 Transfer Learning
model = models.resnet18(weights='DEFAULT')
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 5) # Outputting to our 5 classes
model = model.to(DEVICE)

# 6. Training Logic
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001) # Low LR for fine-tuning

print("Starting Training...")
for epoch in range(5): # 5 Epochs is enough for a strong start
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/5 - Loss: {running_loss/len(train_loader):.4f}")

# 7. Final Evaluation
print("\nEvaluating Model...")
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# 8. Metrics Report
print("\n--- Final Metrics ---")
print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
print(f"F1-Score (Weighted): {f1_score(y_true, y_pred, average='weighted'):.4f}")
print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=CLASS_MAP.keys()))

# 9. Plot Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=CLASS_MAP.keys(), yticklabels=CLASS_MAP.keys())
plt.title("Confusion Matrix: Delhi-NCR Land Use")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()