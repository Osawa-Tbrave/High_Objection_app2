# 以下を「model.py」に書き込み
import torch
# pytorchで画像を扱う torchvisonとデータ整形のtransforms
from torchvision import models, transforms
from PIL import Image

net = models.resnet101(pretrained=True)  # 訓練済みのモデルを読み込み
with open("imagenet_classes.txt") as f:  # ラベルの読み込み
    classes = [line.strip() for line in f.readlines()]

# PIL形式の画像を引き数として受け取る。入力画像の変形をして、Tensor 標準化を施す。
def predict(img):
    # 以下の設定はこちらを参考に設定: https://pytorch.org/hub/pytorch_vision_resnet/
    transform = transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]
                                        )
                                    ])

    # モデルへの入力
    img = transform(img)
    x = torch.unsqueeze(img, 0)  # 形式として次元を増やす。バッチ対応の為。今回は使用しないけど。

    # 予測モード
    net.eval()
    y = net(x)

    # 結果を返す
    y_prob = torch.nn.functional.softmax(torch.squeeze(y))  # 確率で表す。合計値が1になるように。ソフトマックス関数。
    sorted_prob, sorted_indices = torch.sort(y_prob, descending=True)  # 降順にソート
    return [(classes[idx], prob.item()) for idx, prob in zip(sorted_indices, sorted_prob)]
