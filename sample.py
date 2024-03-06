import cv2
import numpy as np

# 顔検出用のカスケード分類器をロード
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# クマの耳の画像を読み込む
ears_img = cv2.imread("path_to_bear_ears.png", -1)  # 透明度を保持するため-1を指定

# 加工したい写真を読み込む
img = cv2.imread("path_to_photo.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 顔検出
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for x, y, w, h in faces:
    # クマの耳のリサイズ（顔の幅に合わせる）
    ears_width = w
    ears_height = int(ears_img.shape[0] * (ears_width / ears_img.shape[1]))
    ears_resized = cv2.resize(ears_img, (ears_width, ears_height))

    # クマの耳を顔の上に配置するための位置を計算（耳の下部が顔の上部に合うように）
    ears_x1 = x
    ears_x2 = ears_x1 + ears_width
    ears_y1 = y - ears_height
    ears_y2 = y

    # クマの耳を写真に合成
    for i in range(0, ears_height):
        for j in range(0, ears_width):
            if ears_resized[i, j][3] != 0:  # 透明度が0ではないピクセルのみ
                img[ears_y1 + i, ears_x1 + j] = ears_resized[i, j][:3]

# 加工後の画像を表示
cv2.imshow("Image with Bear Ears", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
