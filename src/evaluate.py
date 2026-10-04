import torch
from dataset import BirdDroneDataset
from model_cnn import SimpleCNN
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

test_image_dir = 'data/test/images'
test_label_dir = 'data/test/labels'
img_size = 224
test_tfm = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_dataset = BirdDroneDataset(test_image_dir, test_label_dir, transform=test_tfm)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
model = SimpleCNN()
model.load_state_dict(torch.load("bird_drone_best.pth"))
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        predictions = outputs.argmax(dim=1)
        y_true.extend(labels.numpy())
        y_pred.extend(predictions.numpy())
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, target_names=['Bird', 'Drone']))




















# import matplotlib.pyplot as plt
# from sklearn.metrics import classification_report, confusion_matrix
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# def evaluate_model(model_path):
#     val_datagen = ImageDataGenerator(rescale=1./255)
#     test_gen = val_datagen.flow_from_directory(
#         'data/test',
#         target_size=(224,224),
#         batch_size=32,
#         class_mode='binary',
#         shuffle=False
#     )
#     model = load_model(model_path)
#     preds = (model.predict(test_gen) > 0.5).astype(int)
#     true_classes = test_gen.classes
#     class_labels = list(test_gen.class_indices.keys())
#     print(classification_report(true_classes, preds, target_names=class_labels))

#     cm = confusion_matrix(true_classes, preds)
#     plt.imshow(cm, cmap=plt.cm.Blues)
#     plt.xlabel("Predicted")
#     plt.ylabel("True")
#     plt.title("Confusion Matrix")
#     plt.colorbar()
#     plt.show()

# if __name__ == "__main__":
#     evaluate_model('models/best_custom_cnn.h5')










































# import matplotlib.pyplot as plt
# from sklearn.metrics import classification_report, confusion_matrix
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# def evaluate_model(model_path):
#     model = load_model(model_path)

#     val_datagen = ImageDataGenerator(rescale=1./255)
#     val_gen = val_datagen.flow_from_directory(
#         'data',
#         target_size=(224,224),
#         batch_size=32,
#         shuffle=False,
#         class_mode='binary'
#     )

#     val_gen.reset()
#     preds = model.predict(val_gen)
#     preds = (preds > 0.5).astype(int)

#     true_classes = val_gen.classes
#     class_labels = list(val_gen.class_indices.keys())

#     print(classification_report(true_classes, preds, target_names=class_labels))
#     cm = confusion_matrix(true_classes, preds)

#     plt.figure(figsize=(6,6))
#     plt.imshow(cm, cmap=plt.cm.Blues)
#     plt.title('Confusion Matrix')
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.colorbar()
#     plt.show()

# if __name__ == "__main__":
#     evaluate_model('models/best_custom_cnn.h5')
