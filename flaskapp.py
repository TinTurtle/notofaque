
from flask import Flask, render_template, request, jsonify
from flask import Flask, render_template, request
import cv2
from torchvision import transforms
from werkzeug.utils import secure_filename
# from model import Model  
# from Predict.ipynb import predict 
import torch
import numpy as np

from torch import nn
from torchvision import models
import torch.nn.functional as F
import matplotlib.pyplot as plt
import face_recognition
import os
class Model(nn.Module):
    def __init__(
        self,
        num_classes,
        latent_dim=2048,
        lstm_layers=1,
        hidden_dim=2048,
        bidirectional=False,
    ):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        if x.dim() == 4:
       
            x = x.unsqueeze(0).unsqueeze(1)
        
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))

import torch


def inv_normalize(image):
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

   
    image = torch.tensor(image)

   
    if len(image.shape) == 4:
   
        image = image * torch.tensor(std).reshape(1, -1, 1, 1) + torch.tensor(
            mean
        ).reshape(1, -1, 1, 1)
    elif len(image.shape) == 3:
       
        image = image * torch.tensor(std).reshape(-1, 1, 1) + torch.tensor(
            mean
        ).reshape(-1, 1, 1)
    else:
        raise ValueError("Unsupported image shape")

    return image.numpy()


def im_convert(tensor):
    image = tensor.cpu().clone().detach().numpy()
    image = image.squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array([0.5, 0.5, 0.5]) + np.array([0.5, 0.5, 0.5])
    image = image.clip(0, 1)
    return image


app = Flask(__name__)
def predict(model, img, path="./"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fmap, logits = model(img.to("cuda"))
    params = list(model.parameters())
    weight_softmax = model.linear1.weight.detach().cpu().numpy()
    logits = F.softmax(logits, dim=1)
    _, prediction = torch.max(logits, 1)
    confidence = logits[:, int(prediction.item())].item() * 100
    print("confidence of prediction:", logits[:, int(prediction.item())].item() * 100)
    idx = np.argmax(logits.detach().cpu().numpy())
    bz, nc, h, w = fmap.shape
    out = np.dot(
        fmap[-1].detach().cpu().numpy().reshape((nc, h * w)).T, weight_softmax[idx, :].T
    )
    predict = out.reshape(h, w)
    predict = predict - np.min(predict)
    predict_img = predict / np.max(predict)
    predict_img = np.uint8(255 * predict_img)
    out = cv2.resize(predict_img, (im_size, im_size))
    heatmap = cv2.applyColorMap(out, cv2.COLORMAP_JET)
    img = im_convert(img[:, -1, :, :, :])
    result_image = heatmap * 0.5 + img * 0.8 * 255
    cv2.imwrite("/content/1.png", result_image)
    result_image = heatmap * 0.5 / 255 + img * 0.8
    r, g, b = cv2.split(result_image)
    result_image = cv2.merge((r, g, b))

    return {
        "result_image": result_image,
        "result": int(prediction.item()),
        "confidence": confidence,
    }



model = Model(2).cuda()  
path_to_model = "./assets/model_93_acc_100_frames_celeb_FF_data.pt"
model.load_state_dict(torch.load(path_to_model))
model.eval()


im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)


temp_dir = "./temp"
os.makedirs(temp_dir, exist_ok=True)


from flask import Flask, render_template, request
from torchvision import transforms
import cv2
import torch
import numpy as np
from torch import nn
from torchvision import models
import torch.nn.functional as F
import os
import tempfile

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")





@app.route("/predict", methods=["GET","POST"])
def predict_route():
    if request.method == "POST":
        file = request.files["file"]

        file_path = "./temp/temp_file"
        file.save(file_path)

        # Perform prediction based on file type
        if file.content_type.startswith("image"):
           
            img = transform(cv2.imread(file_path))
            img = img.unsqueeze(0).unsqueeze(1).cuda()
            prediction_result = predict(model, img, path="./")
        elif file.content_type.startswith("video"):
           
            video_frames = []
            cap = cv2.VideoCapture(file_path)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                img = transform(frame)
                img = img.unsqueeze(0).unsqueeze(1).cuda()
                prediction_result = predict(model, img, path="./")
                video_frames.append(prediction_result["result_image"])

            cap.release()
            cv2.destroyAllWindows()

          
            result_video_path = "./temp/result_video.avi"
            height, width, layers = video_frames[0].shape
            video = cv2.VideoWriter(
                result_video_path, cv2.VideoWriter_fourcc(*"DIVX"), 1, (width, height)
            )
            for frame in video_frames:
                video.write(frame)

            video.release()
            prediction_result["result_image"] = result_video_path
        else:
            return render_template("index.html", result="Unsupported file type")

       
        if prediction_result["result"] == 1:
            result = "REAL"
        else:
            result = "FAKE"

        
        prediction_result = {"result": result}

    
        return jsonify(prediction_result)


if __name__ == "__main__":
    app.run(debug=True)
