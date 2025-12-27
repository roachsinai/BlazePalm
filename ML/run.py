import os
import torch
import cv2 as cv
import numpy as np

from blazepalm import PalmDetector
from handlandmarks import HandLandmarks

INPUT_SIZE = 256

m = PalmDetector()
m.load_weights("./palmdetector.pth")
m.load_anchors("./anchors.npy")

hl = HandLandmarks()
hl.load_weights("./handlandmarks.pth")

cap = cv.VideoCapture(0)
while True:
    key = cv.waitKey(10)
    if key == 27:  # ESC
        break
    ret, frame = cap.read()
    frame = cv.flip(frame, 1)

    fheight, fwidth, _ = frame.shape
    h_scale, w_scale = fheight / INPUT_SIZE, fwidth / INPUT_SIZE
    img = cv.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    predictions = m.predict_on_image(img)
    if key != -1:
        print(predictions)
    for batch_preds in predictions:
        for pred in batch_preds:
            coords = pred[: -1]
            coords *= INPUT_SIZE
            coords[::2] *= w_scale
            coords[1::2] *= h_scale
            coords = coords.type(torch.IntTensor)
            # crop this image, pad it, run landmarks
            x = max(0, coords[0])
            y = max(0, coords[1])
            endx = min(fwidth, coords[2])
            endy = min(fheight, coords[3])
            # cropped_hand = frame[y:endy, x:endx]
            # maxl = max(cropped_hand.shape[0], cropped_hand.shape[1])
            # cropped_hand = np.pad(cropped_hand,
            #     ( ((maxl-cropped_hand.shape[0])//2, (maxl-cropped_hand.shape[0]+1)//2), ((maxl-cropped_hand.shape[1])//2, (maxl-cropped_hand.shape[1]+1)//2), (0, 0) ),
            #     'constant')
            cv.rectangle(frame, (int(x), int(y)), (int(endx), int(endy)), (255, 0, 0), 2)
            # cv.imshow('img', img)
            # cropped_hand = cv.resize(cropped_hand, (256, 256))
            # _, _, landmarks = hl(torch.from_numpy(cropped_hand).permute((2, 0, 1)).unsqueeze(0))
            # print(landmarks)
            # cv.rectangle(img, (p[0], p[1]), (p[2], p[3]), (255, 0, 0), 2)
            print(f"Confidence: {pred[-1]:4f}", end='\r')

            cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
