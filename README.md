# DL-Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional neural network (CNN) classification model for the given dataset.

## THEORY
The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each of size 28×28 pixels. The task is to classify these images into their respective digit categories. CNNs are particularly well-suited for image classification tasks as they can automatically learn spatial hierarchies of features through convolutional layers, pooling layers, and fully connected layers.

## Neural Network Model
![Uploading nn.svg<svg xmlns="http://www.w3.org/2000/svg" width="2045" height="990" style="cursor: move;"><g><rect class="rect" id="0_0" width="128" height="128" x="510.5" y="403" style="fill: rgb(160, 160, 160); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="0_1" width="128" height="128" x="518.5" y="411" style="fill: rgb(224, 224, 224); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="0_2" width="128" height="128" x="526.5" y="419" style="fill: rgb(160, 160, 160); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="0_3" width="128" height="128" x="534.5" y="427" style="fill: rgb(224, 224, 224); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="0_4" width="128" height="128" x="542.5" y="435" style="fill: rgb(160, 160, 160); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="0_5" width="128" height="128" x="550.5" y="443" style="fill: rgb(224, 224, 224); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="0_6" width="128" height="128" x="558.5" y="451" style="fill: rgb(160, 160, 160); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="0_7" width="128" height="128" x="566.5" y="459" style="fill: rgb(224, 224, 224); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="1_0" width="64" height="64" x="734.5" y="435" style="fill: rgb(160, 160, 160); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="1_1" width="64" height="64" x="742.5" y="443" style="fill: rgb(224, 224, 224); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="1_2" width="64" height="64" x="750.5" y="451" style="fill: rgb(160, 160, 160); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="1_3" width="64" height="64" x="758.5" y="459" style="fill: rgb(224, 224, 224); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="1_4" width="64" height="64" x="766.5" y="467" style="fill: rgb(160, 160, 160); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="1_5" width="64" height="64" x="774.5" y="475" style="fill: rgb(224, 224, 224); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="1_6" width="64" height="64" x="782.5" y="483" style="fill: rgb(160, 160, 160); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="1_7" width="64" height="64" x="790.5" y="491" style="fill: rgb(224, 224, 224); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="2_0" width="48" height="48" x="864.5" y="379" style="fill: rgb(160, 160, 160); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="2_1" width="48" height="48" x="872.5" y="387" style="fill: rgb(224, 224, 224); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="2_2" width="48" height="48" x="880.5" y="395" style="fill: rgb(160, 160, 160); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="2_3" width="48" height="48" x="888.5" y="403" style="fill: rgb(224, 224, 224); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="2_4" width="48" height="48" x="896.5" y="411" style="fill: rgb(160, 160, 160); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="2_5" width="48" height="48" x="904.5" y="419" style="fill: rgb(224, 224, 224); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="2_6" width="48" height="48" x="912.5" y="427" style="fill: rgb(160, 160, 160); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="2_7" width="48" height="48" x="920.5" y="435" style="fill: rgb(224, 224, 224); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="2_8" width="48" height="48" x="928.5" y="443" style="fill: rgb(160, 160, 160); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="2_9" width="48" height="48" x="936.5" y="451" style="fill: rgb(224, 224, 224); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="2_10" width="48" height="48" x="944.5" y="459" style="fill: rgb(160, 160, 160); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="2_11" width="48" height="48" x="952.5" y="467" style="fill: rgb(224, 224, 224); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="2_12" width="48" height="48" x="960.5" y="475" style="fill: rgb(160, 160, 160); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="2_13" width="48" height="48" x="968.5" y="483" style="fill: rgb(224, 224, 224); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="2_14" width="48" height="48" x="976.5" y="491" style="fill: rgb(160, 160, 160); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="2_15" width="48" height="48" x="984.5" y="499" style="fill: rgb(224, 224, 224); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="2_16" width="48" height="48" x="992.5" y="507" style="fill: rgb(160, 160, 160); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="2_17" width="48" height="48" x="1000.5" y="515" style="fill: rgb(224, 224, 224); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="2_18" width="48" height="48" x="1008.5" y="523" style="fill: rgb(160, 160, 160); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="2_19" width="48" height="48" x="1016.5" y="531" style="fill: rgb(224, 224, 224); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="2_20" width="48" height="48" x="1024.5" y="539" style="fill: rgb(160, 160, 160); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="2_21" width="48" height="48" x="1032.5" y="547" style="fill: rgb(224, 224, 224); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="2_22" width="48" height="48" x="1040.5" y="555" style="fill: rgb(160, 160, 160); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="2_23" width="48" height="48" x="1048.5" y="563" style="fill: rgb(224, 224, 224); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="3_0" width="16" height="16" x="1076.5" y="395" style="fill: rgb(160, 160, 160); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="3_1" width="16" height="16" x="1084.5" y="403" style="fill: rgb(224, 224, 224); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="3_2" width="16" height="16" x="1092.5" y="411" style="fill: rgb(160, 160, 160); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="3_3" width="16" height="16" x="1100.5" y="419" style="fill: rgb(224, 224, 224); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="3_4" width="16" height="16" x="1108.5" y="427" style="fill: rgb(160, 160, 160); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="3_5" width="16" height="16" x="1116.5" y="435" style="fill: rgb(224, 224, 224); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="3_6" width="16" height="16" x="1124.5" y="443" style="fill: rgb(160, 160, 160); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="3_7" width="16" height="16" x="1132.5" y="451" style="fill: rgb(224, 224, 224); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="3_8" width="16" height="16" x="1140.5" y="459" style="fill: rgb(160, 160, 160); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="3_9" width="16" height="16" x="1148.5" y="467" style="fill: rgb(224, 224, 224); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="3_10" width="16" height="16" x="1156.5" y="475" style="fill: rgb(160, 160, 160); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="3_11" width="16" height="16" x="1164.5" y="483" style="fill: rgb(224, 224, 224); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="3_12" width="16" height="16" x="1172.5" y="491" style="fill: rgb(160, 160, 160); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="3_13" width="16" height="16" x="1180.5" y="499" style="fill: rgb(224, 224, 224); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="3_14" width="16" height="16" x="1188.5" y="507" style="fill: rgb(160, 160, 160); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="3_15" width="16" height="16" x="1196.5" y="515" style="fill: rgb(224, 224, 224); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="3_16" width="16" height="16" x="1204.5" y="523" style="fill: rgb(160, 160, 160); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="3_17" width="16" height="16" x="1212.5" y="531" style="fill: rgb(224, 224, 224); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="3_18" width="16" height="16" x="1220.5" y="539" style="fill: rgb(160, 160, 160); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="3_19" width="16" height="16" x="1228.5" y="547" style="fill: rgb(224, 224, 224); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="3_20" width="16" height="16" x="1236.5" y="555" style="fill: rgb(160, 160, 160); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="3_21" width="16" height="16" x="1244.5" y="563" style="fill: rgb(224, 224, 224); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="3_22" width="16" height="16" x="1252.5" y="571" style="fill: rgb(160, 160, 160); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="rect" id="3_23" width="16" height="16" x="1260.5" y="579" style="fill: rgb(224, 224, 224); stroke: black; stroke-width: 1; opacity: 0.8;"></rect><rect class="conv" id="conv_0" width="8" height="8" x="654.5122556201801" y="486.8133187819104" style="fill-opacity: 0; stroke: black; stroke-width: 1; stroke-opacity: 0.8;"></rect><rect class="conv" id="conv_1" width="16" height="16" x="826.778943752054" y="533.6418397370891" style="fill-opacity: 0; stroke: black; stroke-width: 1; stroke-opacity: 0.8;"></rect><rect class="conv" id="conv_2" width="8" height="8" x="1083.3604921620079" y="584.1756649577819" style="fill-opacity: 0; stroke: black; stroke-width: 1; stroke-opacity: 0.8;"></rect><line class="link" id="conv_0" x1="662.5122556201801" y1="494.8133187819104" x2="837.439869664096" y2="505.8337700170189" style="stroke: black; stroke-width: 0.5; stroke-opacity: 0.8;"></line><line class="link" id="conv_0" x1="662.5122556201801" y1="486.8133187819104" x2="837.439869664096" y2="505.8337700170189" style="stroke: black; stroke-width: 0.5; stroke-opacity: 0.8;"></line><line class="link" id="conv_1" x1="842.778943752054" y1="549.6418397370891" x2="1084.7789437520541" y2="605.6418397370891" style="stroke: black; stroke-width: 0.5; stroke-opacity: 0.8;"></line><line class="link" id="conv_1" x1="842.778943752054" y1="533.6418397370891" x2="1084.7789437520541" y2="605.6418397370891" style="stroke: black; stroke-width: 0.5; stroke-opacity: 0.8;"></line><line class="link" id="conv_2" x1="1091.3604921620079" y1="592.1756649577819" x2="1274.4441968648032" y2="587.4702659831128" style="stroke: black; stroke-width: 0.5; stroke-opacity: 0.8;"></line><line class="link" id="conv_2" x1="1091.3604921620079" y1="584.1756649577819" x2="1274.4441968648032" y2="587.4702659831128" style="stroke: black; stroke-width: 0.5; stroke-opacity: 0.8;"></line><polygon class="poly" id="fc_0" points="1256.5,367 1266.5,367 1522.5,623 1512.5,623" style="fill: rgb(224, 224, 224); stroke: black; stroke-width: 1; opacity: 0.8;"></polygon><polygon class="poly" id="fc_1" points="1524.5,431 1534.5,431 1662.5,559 1652.5,559" style="fill: rgb(224, 224, 224); stroke: black; stroke-width: 1; opacity: 0.8;"></polygon><line class="line" id="fc_0" x1="1276.5" y1="595" x2="1512.5" y2="623" style="stroke: black; stroke-width: 0.5; stroke-opacity: 0.8; opacity: 1;"></line><line class="line" id="fc_0" x1="1092.5" y1="395" x2="1256.5" y2="367" style="stroke: black; stroke-width: 0.5; stroke-opacity: 0.8; opacity: 1;"></line><line class="line" id="fc_1" x1="1522.5" y1="623" x2="1652.5" y2="559" style="stroke: black; stroke-width: 0.5; stroke-opacity: 0.8; opacity: 1;"></line><line class="line" id="fc_1" x1="1266.5" y1="367" x2="1524.5" y2="431" style="stroke: black; stroke-width: 0.5; stroke-opacity: 0.8; opacity: 1;"></line><text class="text" dy=".35em" font-family="sans-serif" x="729.5" y="659" style="font-size: 16px; opacity: 1;">Max-Pool</text><text class="text" dy=".35em" font-family="sans-serif" x="902.5" y="659" style="font-size: 16px; opacity: 1;">Convolution</text><text class="text" dy=".35em" font-family="sans-serif" x="1121.5" y="659" style="font-size: 16px; opacity: 1;">Max-Pool</text><text class="text" dy=".35em" font-family="sans-serif" x="1315.5" y="659" style="font-size: 16px; opacity: 1;">Dense</text><text class="info" dy="-0.3em" font-family="sans-serif" x="510.5" y="388" style="font-size: 16px;">8@128x128</text><text class="info" dy="-0.3em" font-family="sans-serif" x="734.5" y="420" style="font-size: 16px;">8@64x64</text><text class="info" dy="-0.3em" font-family="sans-serif" x="864.5" y="364" style="font-size: 16px;">24@48x48</text><text class="info" dy="-0.3em" font-family="sans-serif" x="1076.5" y="380" style="font-size: 16px;">24@16x16</text><text class="info" dy="-0.3em" font-family="sans-serif" x="1256.5" y="352" style="font-size: 16px;">1x256</text><text class="info" dy="-0.3em" font-family="sans-serif" x="1524.5" y="416" style="font-size: 16px;">1x128</text></g></svg>…]()



## DESIGN STEPS
### STEP 1: 
Import all the required libraries (PyTorch, TorchVision, NumPy, Matplotlib, etc.).

### STEP 2: 
Download and preprocess the MNIST dataset using transforms.

### STEP 3: 
Create a CNN model with convolution, pooling, and fully connected layers.

### STEP 4: 
Set the loss function and optimizer. Move the model to GPU if available.

### STEP 5: 
Train the model using the training dataset for multiple epochs.

### STEP 6: 
Evaluate the model using the test dataset and visualize the results (accuracy, confusion matrix, classification report, sample prediction).

## PROGRAM
### Name: HAREVASU S
### Register Number: 212223230069

```python
import torch as t
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
train_dataset=torchvision.datasets.MNIST(root='./data',train=True,download=True,transform=transform)
test_dataset=torchvision.datasets.MNIST(root='./data',train=False,download=True,transform=transform)

image,label=train_dataset[0]
print("Image shape:",image.shape)
print("Number of training samples:",len(train_dataset))

image,label=test_dataset[0]
print("Image shape:",image.shape)
print("Number of testing samples:",len(test_dataset))
train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=32,shuffle=False)

class CNNClassifier(nn.Module):
  def __init__(self):
    super(CNNClassifier,self).__init__()
    self.conv1=nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
    self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
    self.conv3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
    self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
    self.fc1=nn.Linear(128*3*3,128)
    self.fc2=nn.Linear(128,64)
    self.fc3=nn.Linear(64,10)

  def forward(self,x):
    x=self.pool(t.relu(self.conv1(x)))
    x=self.pool(t.relu(self.conv2(x)))
    x=self.pool(t.relu(self.conv3(x)))
    x=x.view(x.size(0),-1)
    x=nn.functional.relu(self.fc1(x))
    x=nn.functional.relu(self.fc2(x))
    x=self.fc3(x)
    return x

from torchsummary import summary
model=CNNClassifier()
if t.cuda.is_available():
  device=t.device("cuda")
  model.to(device)
print("Name: HAREVASU S")
print("Reg.no: 212223230069")
summary(model,input_size=(1,28,28))
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)
def train_model(model,train_loader,num_epochs):
  for epoch in range(num_epochs):
    model.train()
    running_loss=0.0
    for images,labels in train_loader:
      if t.cuda.is_available():
        images,labels=images.to(device),labels.to(device)
      optimizer.zero_grad()
      outputs=model(images)
      loss=criterion(outputs,labels)
      loss.backward()
      optimizer.step()
      running_loss+=loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
print("Name: HAREVASU S")
print("Reg.no: 212223230069")

train_model(model,train_loader,num_epochs=10)

def test_model(model, test_loader):
  model.eval()
  correct = 0
  total = 0
  all_preds = []
  all_labels = []
  with t.no_grad():
    for images, labels in test_loader:
      if t.cuda.is_available():
        images, labels = images.to(device), labels.to(device)

      outputs = model(images)
      _, predicted = t.max(outputs, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
      all_preds.extend(predicted.cpu().numpy())
      all_labels.extend(labels.cpu().numpy())

  accuracy = correct/total
  print("Name: HAREVASU S")
  print("Reg.no: 212223230069")
  print(f"Test Accuracy: {accuracy:.4f}")

  cm = confusion_matrix(all_labels, all_preds)
  plt.figure(figsize=(8, 6))
  print("Name: HAREVASU S")
  print("Reg.no: 212223230069")
  sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
  plt.xlabel("Predicted")
  plt.ylabel("Actual")
  plt.title("Confusion Matrix")
  plt.show()

  print("Name: HAREVASU S")
  print("Reg.no: 212223230069")
  print("Classification Report:")
  print(classification_report(all_labels, all_preds, target_names=[str(i) for i in range(10)]))
test_model(model, test_loader)

def predict_image(model,image_index,dataset):
  model.eval()
  image,label=dataset[image_index]
  if t.cuda.is_available():
    image=image.to(device)

  with t.no_grad():
    output=model(image.unsqueeze(0))
    _,predicted=t.max(output,1)
  class_names=[str(i) for i in range(10)]
  print("Name: HAREVASU S")
  print("Reg.no: 212223230069")
  plt.imshow(image.cpu().squeeze(0),cmap='gray')
  plt.title(f"Actual: {class_names[label]}\nPredicted: {class_names[predicted.item()]}")
  plt.axis("off")
  plt.show()
  print(f"Actual: {class_names[label]}\nPredicted: {class_names[predicted.item()]}")
predict_image(model,image_index=80,dataset=test_dataset)

```
### OUTPUT

## Training Loss per Epoch
![image](https://github.com/user-attachments/assets/88e5e9ab-669a-4c8e-9387-4677837f3f37)


## Confusion Matrix
![image](https://github.com/user-attachments/assets/bf220126-fbd5-4788-b563-f635be084f3d)



## Classification Report
![image](https://github.com/user-attachments/assets/274f5842-0d80-4853-9f81-a2b278c925bf)


### New Sample Data Prediction
![image](https://github.com/user-attachments/assets/abf0f621-102c-4f2d-b46a-b630d8c02332)


## RESULT
Thus the CNN model was trained and tested successfully on the MNIST dataset.
