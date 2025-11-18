import time 
import cv2
import threading


class VideoPlayer:
    def __init__(self, source, width=1280, height=720):
        self.cv2 = cv2
        self.__cap = cv2.VideoCapture(source)
        if not self.__cap.isOpened():
            print(f"Error: Cannot open {'camera' if isinstance(source, int) else ''} {source}")
            self.__cap.release()
        _, self.__frame = self.__cap.read()
        self.__lock = threading.Lock()
        self.__thread = None
        self.__stop = False
        self.__last_read_time = time.time()

    def start(self):
        self.__stop = False
        self.__thread = threading.Thread(target=self.__run, daemon=True)
        self.__thread.start()

    def stop(self):
        self.__stop = True
        if self.__thread is not None:
            self.__thread.join()
        self.__cap.release()

    def __run(self):
        while not self.__stop:
            ret, frame = self.__cap.read()
            if not ret:
                print("Warning: Lost connection to the camera stream.")
                time.sleep(1)
                break
            self.__last_read_time = time.time()
            with self.__lock:
                self.__frame = frame
        with self.__lock:
            self.__frame = None

    def is_opened(self):
        with self.__lock:
            if self.__cap is not None:
                return self.__cap.isOpened()
            else:
                return False

    def get_capture(self):
        with self.__lock:
            return self.__cap

    def next(self):
        with self.__lock:
            if self.__frame is not None:
                frame = self.__frame.copy()
            else:
                frame = None
        return frame


class VideoPlayerOffline:
    def __init__(self, source, width=1280, height=720):
        self.cv2 = cv2
        self.__cap = cv2.VideoCapture(source)
        if not self.__cap.isOpened():
            raise RuntimeError(f"Cannot open {'camera' if isinstance(source, int) else ''} {source}")
        self.__frame = None
        self.__stop = False

    def start(self):
        self.__stop = False

    def stop(self):
        self.__stop = True
        self.__cap.release()

    def next(self):
        if self.__stop:
            return None
        ret, frame = self.__cap.read()
        if not ret:
            return None
        return frame.copy()
