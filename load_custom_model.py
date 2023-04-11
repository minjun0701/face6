import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
from math import hypot
import dlib
import time
import winsound

eye_closed = False
closed_time = 0
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
model = torch.hub.load('ultralytics/yolov5', 'custom', path='venv/yolov5/runs/train/exp8/weights/last.pt', force_reload=True)

cap = cv2.VideoCapture(0)

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

font = cv2.FONT_HERSHEY_PLAIN

def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
    hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)
    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
    ratio = hor_line_lenght / ver_line_lenght
    return ratio

while cap.isOpened():
    ret, frame = cap.read()

    # 프레임을 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 얼굴 영역 검출
    faces = detector(gray)

    for face in faces:
        # 얼굴 랜드마크 검출
        landmarks = predictor(gray, face)

        # 왼쪽 눈 영역 좌표 추출
        left_eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                    (landmarks.part(37).x, landmarks.part(37).y),
                                    (landmarks.part(38).x, landmarks.part(38).y),
                                    (landmarks.part(39).x, landmarks.part(39).y),
                                    (landmarks.part(40).x, landmarks.part(40).y),
                                    (landmarks.part(41).x, landmarks.part(41).y)], np.int32)

        # 눈 영역에 경계선 그리기
        cv2.polylines(frame, [left_eye_region], True, (0, 255, 0), 2)

        # 왼쪽 눈 영역 마스크 생성
        mask = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
        cv2.fillPoly(mask, [left_eye_region], 255)

        # 눈 영역 추출
        left_eye = cv2.bitwise_and(gray, gray, mask=mask)

        # 눈 영역의 평균 밝기 구하기
        left_eye_mean = cv2.mean(left_eye)[0]

        # 눈이 감긴 상태인지 체크
        if left_eye_mean < 40:
            if not eye_closed:
                closed_time = time.time()
            eye_closed = True
        else:
            if eye_closed:
                elapsed_time = time.time() - closed_time
                if elapsed_time > 5:  # 눈을 5초 이상 감았다면
                    winsound.Beep(440, 1000)  # 비프음 울리기 (440Hz, 1초)
                eye_closed = False
        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        if blinking_ratio > 7.0:
            winsound.Beep(440, 1000)

    # Make detections
    results = model(frame)

    cv2.imshow('YOLO', np.squeeze(results.render()))

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()