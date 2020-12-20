from PyQt5 import QtCore, QtGui, QtWidgets
import main2
from PIL import Image
import numpy as np
import cv2
import pytesseract
import pyautogui
from pytesseract import Output
import multiprocessing

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class Test(main2.Ui_Foldem):
    def __init__(self):
        self.sorted_hand = ''

    def setupUi2(self, Foldem2):
        super().setupUi(Foldem2)
        self.Update.clicked.connect(self.screengrab)
        self.Recognize.clicked.connect(self.recognize)
        self.decision_qs.clicked.connect(lambda:self.decision(self.sorted_hand))

    def screengrab(self):
        lx = self.LHInputX.text()
        ly = self.LHInputY.text()
        rx = self.RHInputX.text()
        ry = self.RHInputY.text()
        fx = self.FoldX.text()
        fy = self.FoldY.text()
        l_width = self.LHInputX_width.text()
        l_height = self.LHInputY_height.text()
        r_width = self.RHInputX_width.text()
        r_height = self.RHInputY_height.text()
        f_width = self.FoldX_width.text()
        f_height = self.FoldY_height.text()

        left_image = pyautogui.screenshot(region=(lx, ly, l_width, l_height))
        right_image = pyautogui.screenshot(region=(rx, ry, r_width, r_height))
        fold_image = pyautogui.screenshot(region=(fx, fy, f_width, f_height))
        left_image.save(r'Test Images\left.png')
        right_image.save(r'Test Images\right.png')
        fold_image.save(r'Test Images\fold.png')

        #save test images
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
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def thresholding(self, image):
        return cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]

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




    def decision(self,input_hand):
        # decision engine lol
        starting_hand = str('A3')

        def decision_engine(hand):
            Range = ['AA', 'AK', 'AQ', 'AJ', 'AT', 'KK', 'KQ', 'KJ', 'QQ', 'JJ', 'TT', '99', '88', '77', '66', '55',
                     '44', '33', '22']
            return hand in Range

        answer = decision_engine(input_hand)

        self.decision_label.setText(str(answer))
        return answer


    def click(self):
        (x, y) = pyautogui.position()
        # Your automated click
        pyautogui.click(625, 1020)
        # Move back to where the mouse was before click
        pyautogui.moveTo(x, y)





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
