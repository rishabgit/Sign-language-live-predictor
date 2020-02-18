import fastai
from fastai import *
from fastai.vision import *
import pathlib
from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer
import cv2
import sys

# Loading models : 
# Loading asl detection model
path_test = pathlib.Path('cropped_pics')
path_train = pathlib.Path('data/training/')
np.random.seed(6)
data = ImageDataBunch.from_folder(path=path_train, train=".", ds_tfms=get_transforms(),
                                size=224, valid_pct=0.2).normalize(imagenet_stats)
learn = cnn_learner(data, models.resnet34, metrics=accuracy)
learn.load('asl-mixed-2')

# Loading hand detection model 
net_type = 'mb1-ssd' #sys.argv[1]
model_path = 'models/mb1-ssd-Epoch-8-Loss-3.6308415750177896.pth' #sys.argv[2]
label_path = 'models/open-images-model-labels.txt' #sys.argv[3]
class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)
net = create_mobilenetv1_ssd(len(class_names), is_test=True)
net.load(model_path)
predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)


def asl2text(orig_image):
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    boxes, labels, probs = predictor.predict(image, 10, 0.4)
    if boxes.size(0) is 1:
        box = boxes[0, :]
        cropped_image = orig_image[int(box[1])-70:int(box[3])+70,int(box[0])-40:int(box[2])+40]
        path = "cropped_pics/cropped.jpg" # It brings file from path_test 
        cv2.imwrite(path, cropped_image) 
        img = open_image(get_image_files(path_test)[0])
        pred_class,pred_idx,outputs = learn.predict(img)
        os.remove(path)
        print(pred_class)
        #return box, pred_class
        return int(box[0]), int(box[1]), int(box[2]), int(box[3]), pred_class
    #return 'Nothing','Nothing'
    return 'Nothing', 'Nothing', 'Nothing', 'Nothing', 'Nothing'
















from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QHBoxLayout, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import QtCore
from PyQt5.QtCore import QTimer
from PyQt5 import uic
import sys
import cv2
from GUI_utils import init, audio2text, text2native
#from asl_to_text import *


class Ui(QWidget):
    def __init__(self):
        super().__init__()
        ui = uic.loadUi('final_ui.ui', self)
        self.string = ''
        self.language = 29
        self.data= init()
        print(self.data)


    ########################################     1st Page Function      ##############################################
        self.Voice.clicked.connect(lambda : self.changepage(0,1))
        self.Gesutre.clicked.connect(lambda : self.changepage(0,2))
        self.New_Data.clicked.connect(lambda : self.changepage(0,3))
        for i in self.data:
            self.Language_page1.addItem("")
            self.Language_page1.setItemText(i,self.data[i][0])
        self.Language_page1.setCurrentIndex(29)

        self.Language_page1.view().setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)


    ########################################     2nd Page Function  (Audio)     ##############################################
        self.Back_page2.clicked.connect(lambda : self.changepage(1,0))

        for i in self.data:
            self.Language_page2.addItem("")
            self.Language_page2.setItemText(i,self.data[i][0])
        self.Language_page2.setCurrentIndex(29)

        self.Audio_page2.clicked.connect(lambda : self.audio(self.data[self.Language_page2.currentIndex()]))


    ########################################     3nd Page Function (Video)       ##############################################
        self.call = 0     # For evoking the model

        for i in self.data:
            self.Language_page3.addItem("")
            self.Language_page3.setItemText(i,self.data[i][0])
        self.Language_page3.setCurrentIndex(29)

        self.Back_page3.clicked.connect(lambda : self.changepage(2,0))
        self.timer = QTimer()
        self.timer.timeout.connect(self.Cam)
        self.Start_Stop_page3.clicked.connect(lambda: self.controlTimer(0))
        self.Refresh_page3.clicked.connect(self.Refresh)

        ########################################     4th Page Function (Video)       #############################################
        self.Back_page4.clicked.connect(lambda : self.changepage(2,0))
        self.timer1 = QTimer()
        self.timer1.timeout.connect(self.Cam1)
        self.Captue_Page4.clicked.connect(lambda: self.controlTimer(1))




        ####################################           Cam Frame Def       ################################################
        self.Videoframe_page3.setScaledContents(True)
        self.Videoframe_page4.setScaledContents(True)
        PNG = QPixmap(640, 480).toImage()
        self.Videoframe_page3.setPixmap(QPixmap.fromImage(PNG))
        self.Videoframe_page4.setPixmap(QPixmap.fromImage(PNG))




    ##########################################         Audio Page Button Click    ########################################
    def audio(self, lan):
        try:
            self.boxdelete()
        except:
            pass
        print(lan[1:])
        self.audio_text, self.native = audio2text(lan[1:])                                  ####   changed
        self.Dispaly_text_page2.setText(lan[0]+": "+self.native+"\n"+"English: "+self.audio_text)   
        self.addpic(self.audio_text)

    ##############################################      Change Page       ##############################################
    def changepage(self, par, i):
        if(par == 0):
            self.language = self.Language_page1.currentIndex()
            self.Language_page2.setCurrentIndex(self.language)
            self.Language_page3.setCurrentIndex(self.language)
        if(par == 1):
            self.language = self.Language_page2.currentIndex()
            self.Language_page1.setCurrentIndex(self.language)
            self.Language_page3.setCurrentIndex(self.language)
            try:                                                                            ###### Changed Code
                self.Dispaly_text_page2.setText('')                                           ##
                self.boxdelete()                                                                ##
            except:                                                                             ##
                pass                                                                             ###
        if(par == 2):
            self.language = self.Language_page3.currentIndex()
            self.Language_page1.setCurrentIndex(self.language)
            self.Language_page2.setCurrentIndex(self.language)

        self.StackPage.setCurrentIndex(i)


    ##############################################         Cam Util         ###########################################

    def controlTimer(self, control):
        if control == 0:
            #self.string = []
            if not self.timer.isActive():
                print(self.string)
                self.cap = cv2.VideoCapture(0)
                self.timer.start(20)
                self.Start_Stop_page3.setText("Stop")
            else:
                self.timer.stop()
                self.cap.release()
                print('hi')
                print(self.string)
                self.language = self.Language_page3.currentIndex()
                self.Speak_page3.clicked.connect(lambda: self.Audio_gen(self.language))
                print(self.data[self.language][1:])
                self.Predicted_page3.setText(self.string)
                self.Start_Stop_page3.setText("Start")
                self.Videoframe_page3.setScaledContents(True)
                self.call = 1
                PNG = QPixmap(640, 480).toImage()
                self.Videoframe_page3.setPixmap(QPixmap.fromImage(PNG))
                print(self.string)
        if control == 1:
            
            if not self.timer1.isActive():
                if self.Classname_page4.toPlainText() == '':
                    self.Classname_page4.setPlainText("Please Enter")
                    return
                self.cap = cv2.VideoCapture(0)
                self.timer1.start(20)
                self.Start_Stop_page3.setText("Stop")
            else:
                self.timer1.stop()
                self.cap.release()
                self.Start_Stop_page4.setText("Start")
                self.Refresh_page3.clicked.connect(lambda: self.Refresh(0))
                self.Videoframe_page4.setScaledContents(True)
                self.call = 1
                PNG = QPixmap(640, 480).toImage()
                self.Videoframe_page4.setPixmap(QPixmap.fromImage(PNG))

    def Audio_gen(self, lang):
        lang = self.Language_page3.currentIndex()
        text = text2native(self.string, self.data[lang][1:])
        print('run now')
        self.Predicted_page3.setText(text)
    
        

    def Cam(self):
        ret, orig_image = self.cap.read()
        #box0, box1, box2, box3 = -1, -1, -1, -1
        self.call = self.call+1
        print("Printing normal image")
        self.Refresh_page3.clicked.connect(lambda: self.Refresh(0))
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        height, width, channel = image.shape
        step = channel * width
        if(self.call == 50):
            print('reached')
            self.call = 0
            #box, letter = asl2text(orig_image)
            box0, box1, box2, box3, letter = asl2text(orig_image)
            alpha = str(letter)
            if alpha == 'Nothing':
                alpha = ''
            else:
                self.string += alpha
        try:
            if box0 is not 'Nothing':
                print(box0)
                #image = cv2.rectangle(orig_image, (box[0]-40, box[1]-70), (box[2]+40, box[3]+70), (255, 255, 0), 4)
                image = cv2.rectangle(orig_image, (box0-40, box1-70), (box2+40, box3+70), (255, 255, 0), 4)
                #qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
                #self.Videoframe_page3.setPixmap(QPixmap.fromImage(qImg))
                cv2.imshow('bb',image)
            print("Try::")
        except:
            pass

        cv2.imshow('aa',image)
        qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
        self.Videoframe_page3.setScaledContents(True)
        self.Videoframe_page3.setPixmap(QPixmap.fromImage(qImg))

            


    def Cam1(self):
        ret, image = self.cap.read()
        self.call = self.call+1
        import cv2
        import os
        import string
        text = self.Classname_page4.toPlainText()
        text = text.lower()
        path = os.path.join('data\\training',text)
        if os.path.exists(path):
            cv2.imwrite(os.path.join(path, str(len(os.listdir(path)))+'.jpg'), image)
        else:
            os.mkdir(path)
            print(os.path.join(path,'0.jpg'))
            cv2.imwrite(os.path.join(path,'0.jpg'), image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = image.shape
        step = channel * width
        qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
        self.Videoframe_page4.setScaledContents(True)
        self.Videoframe_page4.setPixmap(QPixmap.fromImage(qImg))

        if(self.call == 200):
            print('reached')
            self.call = 0
            self.timer1.stop()
            self.cap.release()
            self.Start_Stop_page3.setText("Start")
            self.Videoframe_page3.setScaledContents(True)
            self.call = 1
            PNG = QPixmap(640, 480).toImage()
            self.Videoframe_page4.setPixmap(QPixmap.fromImage(PNG))

    def Refresh(self, control):
        self.call = 0
        self.Predicted_page3.setText('')
        self.string = ''
        PNG = QPixmap(640, 480).toImage()
        if(control == 0):
            self.Videoframe_page3.setText('')
            self.Videoframe_page3.setScaledContents(True)
            self.Videoframe_page3.setPixmap(QPixmap.fromImage(PNG))
        elif control == 1:
            self.Videoframe_page4.setText('')
            self.Videoframe_page4.setScaledContents(True)
            self.Videoframe_page4.setPixmap(QPixmap.fromImage(PNG))
    ######################################################################       Dynamic addition of Pictures in Screen    ######################################################3

    def addpic(self, text):
        import string
        import os
        letter=list(text.upper())

        content_widget = QWidget()
        self.pic_scrollArea_page2.setWidget(content_widget)
        self.lay = QHBoxLayout(content_widget)
        print('here')
        self.lay.setAlignment(QtCore.Qt.AlignHCenter)
        print('hen')
        for l in letter:
                #print(os.path.join('Sprites', l+'.png'))
                pixmap = QPixmap(os.path.join('Sprites', l+'.png'))
                if not pixmap.isNull():
                    label = QLabel(pixmap=pixmap)
                    self.lay.addWidget(label)
                elif l == ' ':
                    pixmap = QPixmap(os.path.join('Sprites', 'space.png'))
                    if not pixmap.isNull():
                        label = QLabel(pixmap=pixmap)
                        self.lay.addWidget(label)

    ######################################        Delete all qidget inside a layer     ##################################
    def boxdelete(self):
        for i in reversed(range(self.lay.count())):
            self.lay.itemAt(i).widget().deleteLater()



App = QApplication(sys.argv)
window = Ui()
window.show()
sys.exit(App.exec_())
