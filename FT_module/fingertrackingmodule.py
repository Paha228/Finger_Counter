import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands=2, complexity=0, detectionCon=0.75, trackingCon=0.75):
        mpHands = mp.solutions.hands # говорим, что хотим распозновать руки
        self.hands = mpHands.Hands(mode, maxHands, complexity, detectionCon, trackingCon) # характеристики для распознования
        self.mpDraw = mp.solutions.drawing_utils # инициализация утилиты для рисования
        self.fingertips = [4, 8, 12, 16, 20] # кончики пальцев