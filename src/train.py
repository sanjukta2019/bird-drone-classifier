
import torch
import torch.nn as nn
from tqdm import tqdm
import dataset

from torch.optim import Adam
from torch.utils.data import DataLoader
from dataset import BirdDroneDataset
from model_cnn import SimpleCNN
from model_transfer import get_resnet50

from torchvision import transforms

# Paths
train_image_dir = 'data/train/images'
train_label_dir = 'data/train/labels'

# Transforms
img_size = 224
train_tfm = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Dataset and Loader
train_dataset = BirdDroneDataset(train_image_dir, train_label_dir, transform=train_tfm)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Model
model = SimpleCNN()  # or get_resnet50()

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()


    # Validate here
    # Save best model
    # Use EarlyStopping logic
    
num_epochs = 10  # Add this!

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_correct = 0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_correct += (outputs.argmax(dim=1) == labels).sum().item()
    train_acc = train_correct / len(train_loader.dataset)
    print(f"Epoch {epoch+1}, Train loss: {train_loss/len(train_loader)}, Train Acc: {train_acc:.4f}")

# After training, save the model
torch.save(model.state_dict(), "bird_drone_best.pth")
print("Training complete. Model saved to bird_drone_best.pth")















































# from augmentation import get_train_augmentation, get_val_augmentation
# from custom_cnn import build_custom_cnn
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# def train_model():
#     train_datagen = get_train_augmentation()
#     val_datagen = get_val_augmentation()

#     train_gen = train_datagen.flow_from_directory(
#         'data/train/images',
#         target_size=(224,224),
#         batch_size=32,
#         class_mode='binary'
#     )
#     val_gen = val_datagen.flow_from_directory(
#         'data/valid/images',
#         target_size=(224,224),
#         batch_size=32,
#         class_mode='binary'
#     )

#     model = build_custom_cnn()
#     checkpoint = ModelCheckpoint(
#         'models/best_custom_cnn.h5', save_best_only=True, monitor='val_accuracy', mode='max')
#     earlystop = EarlyStopping(monitor='val_loss', patience=7)

#     model.fit(
#         train_gen,
#         validation_data=val_gen,
#         epochs=30,
#         callbacks=[checkpoint, earlystop]
#     )

# if __name__ == "__main__":
#     train_model()































