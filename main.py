from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import *

import main2
from PIL import Image
import numpy as np
import cv2
import pytesseract
import pyautogui
from functools import partial
from pytesseract import Output
import multiprocessing
import pickle
import time
import traceback,sys
from playsound import playsound

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        tuple (exctype, value, traceback.format_exc() )

    result
        object data returned from processing, anything

    progress
        int indicating % progress

    '''
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)


class Worker(QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
    #    self.kwargs['progress_callback'] = self.signals.progress

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done





class Test(main2.Ui_Foldem):
    def __init__(self):
        self.sorted_hand = ''
        self.fx = ''
        self.fy = ''
        self.previous_input = {}
        self.save_input = {'save_left_x': '',
                           'save_left_y': '',
                           'save_left_width': '',
                           'save_left_height': '',
                           'save_right_x': '',
                           'save_right_y': '',
                           'save_right_width': '',
                           'save_right_height': ''
                           }
        self.threadpool = QThreadPool()
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())

    def setupUi2(self, Foldem2):
        super().setupUi(Foldem2)
        self.Update.clicked.connect(self.screengrab)
        self.Recognize.clicked.connect(self.recognize)
        #self.decision_qs.clicked.connect(
            #lambda: self.decision(self.sorted_hand))  # call function within class self.Function
        # self.decision_qs.clicked.connect(self.decision2)
        self.decision_qs.clicked.connect(
            lambda: self.oh_no(self.sorted_hand))  # call function within class self.Function

        # load save inputs
        f = open('save_input.pkl', 'rb')
        self.previous_input = pickle.load(f)
        f.close()
        # TODO: refactor this horrible shit, load values from pickle dictionary to input boxes better
        self.LHInputX.setText(self.previous_input['save_left_x'])
        self.LHInputY.setText(self.previous_input['save_left_y'])
        self.LHInputX_width.setText(self.previous_input['save_left_width'])
        self.LHInputY_height.setText(self.previous_input['save_left_height'])
        self.RHInputX.setText(self.previous_input['save_right_x'])
        self.RHInputY.setText(self.previous_input['save_right_y'])
        self.RHInputX_width.setText(self.previous_input['save_right_width'])
        self.RHInputY_height.setText(self.previous_input['save_right_height'])

    def screengrab(self):
        lx = self.LHInputX.text()
        ly = self.LHInputY.text()
        rx = self.RHInputX.text()
        ry = self.RHInputY.text()
        fx = self.FoldX.text()
        fy = self.FoldY.text()
        self.fx = self.FoldX.text()
        self.fy = self.FoldY.text()
        l_width = self.LHInputX_width.text()
        l_height = self.LHInputY_height.text()
        r_width = self.RHInputX_width.text()
        r_height = self.RHInputY_height.text()
        f_width = self.FoldX_width.text()
        f_height = self.FoldY_height.text()

        # save inputs
        f = open('save_input.pkl', 'wb')
        pickle_save_input = dict(zip(self.save_input, [lx, ly, l_width, l_height, rx, ry, r_width, r_height]))
        pickle.dump(pickle_save_input, f)  # dump data to f
        f.close()

        left_image = pyautogui.screenshot(region=(lx, ly, l_width, l_height))
        right_image = pyautogui.screenshot(region=(rx, ry, r_width, r_height))
        fold_image = pyautogui.screenshot(region=(fx, fy, f_width, f_height))
        left_image.save(r'Test Images\left.png')
        right_image.save(r'Test Images\right.png')
        fold_image.save(r'Test Images\fold.png')

        # save test images
        thresh_left = cv2.imread(r'Test Images\left.png')
        thresh_right = cv2.imread(r'Test Images\right.png')
        thresh_left = self.thresholding(self.get_grayscale(thresh_left))
        thresh_right = self.thresholding(self.get_grayscale(thresh_right))
        cv2.imwrite(r'Test Images\thresh_left.png', thresh_left)
        cv2.imwrite(r'Test Images\thresh_right.png', thresh_right)

        self.lh_card_pic.setPixmap(QtGui.QPixmap(r'Test Images\left.png'))
        self.rh_card_pic.setPixmap(QtGui.QPixmap(r'Test Images\right.png'))
        self.fold_pic.setPixmap(QtGui.QPixmap(r'Test Images\fold.png'))

    # processing
    def get_grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    def thresholding(self, image):
        return cv2.threshold(image, 160, 255, cv2.THRESH_BINARY)[1]

    def recognize(self):
        alphabet = "AKQJT98765432"

        LH_image = cv2.imread(r'Test Images\left.png')
        RH_image = cv2.imread(r'Test Images\right.png')
        # cv2.imshow('test', LH_image)

        LH_card = pytesseract.image_to_string(self.thresholding(self.get_grayscale(LH_image)), lang='eng',
                                              config='--psm 9 -c tessedit_char_whitelist=AKQJ0123456789$')
        RH_card = pytesseract.image_to_string(self.thresholding(self.get_grayscale(RH_image)), lang='eng',
                                              config='--psm 9 -c tessedit_char_whitelist=AKQJ0123456789$')

        self.left_card.setText(LH_card)
        self.right_card.setText(RH_card)

        self.unsorted_hand = [LH_card, RH_card]

        # change 10 to T
        for x in range(len(self.unsorted_hand)):
            if self.unsorted_hand[x] == '10':
                print('ten')
                self.unsorted_hand[x] = 'T'
            elif self.unsorted_hand[x] not in alphabet:
                self.unsorted_hand[x] = ''
                print('yay')
            else:
                pass

        print(self.unsorted_hand[0])
        print(self.unsorted_hand[1])
        # hand sorter

        self.sorted_hand = sorted(self.unsorted_hand, key=lambda word: [alphabet.index(c) for c in word])
        self.sorted_hand = self.sorted_hand[0] + self.sorted_hand[1]

        self.full_hand.setText(self.sorted_hand)

        return self.sorted_hand

    def decision(self, input_hand):
        # decision engine lol
        starting_hand = str('A3')

        def decision_engine(hand):
            Range = ['AA', 'AK', 'AQ', 'AJ', 'AT', 'KK', 'KQ', 'KJ', 'QQ', 'JJ', 'TT', '99', '88', '77', '66', '55',
                     '44', '33', '22']
            return hand in Range

        answer = decision_engine(input_hand)
        while (answer == False):
            self.click()
            QThread.sleep(10)
            answer = decision_engine(input_hand)
            print(f'hand is {answer}: {input_hand}')
        else:
            print('HAND!!!')
            playsound('airhorn1.mp3')

        self.decision_label.setText(str(answer))
        return answer

    def decision2(self):
        self.decision(self.sorted_hand)
        return

    def click(self):
        (x, y) = pyautogui.position()
        # Your automated click
        pyautogui.moveTo(int(self.fx), int(self.fy), 1)
        pyautogui.click(int(self.fx), int(self.fy))
        print('clicked')
        # Move back to where the mouse was before click
        pyautogui.moveTo(x, y)
        return


    #worker tester
    def progress_fn(self, n):
        print("%d%% done" % n)

    def print_output(self, s):
        print(s)

    def thread_complete(self):
        print("THREAD COMPLETE!")

    def oh_no(self, hand):
        # Pass the function to execute
        worker = Worker(self.decision, hand)# Any other args, kwargs are passed to the run function
        worker.signals.result.connect(self.print_output)
        worker.signals.finished.connect(self.thread_complete)
#        worker.signals.progress.connect()

        # Execute
        self.threadpool.start(worker)



    # if __name__ == "__main__":
    #     import sys
    #     app = QtWidgets.QApplication(sys.argv)
    #     Foldem = QtWidgets.QMainWindow()
    #     ui = main2.Ui_Foldem()
    #     ui.setupUi(Foldem)
    #     Foldem.show()
    #     sys.exit(app.exec_())


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    Foldem2 = QtWidgets.QMainWindow()
    ui = Test()
    ui.setupUi2(Foldem2)
    Foldem2.show()
    sys.exit(app.exec_())
