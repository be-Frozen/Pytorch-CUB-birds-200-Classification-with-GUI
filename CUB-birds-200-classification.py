# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'CUB-birds-200-classification.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1115, 750)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.load_Image = QtWidgets.QPushButton(self.centralwidget)
        self.load_Image.setGeometry(QtCore.QRect(570, 80, 501, 41))
        self.load_Image.setObjectName("load_Image")
        self.Title = QtWidgets.QLabel(self.centralwidget)
        self.Title.setGeometry(QtCore.QRect(360, 20, 411, 51))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(17)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        font.setKerning(False)
        self.Title.setFont(font)
        self.Title.setObjectName("Title")
        self.showresult = QtWidgets.QGraphicsView(self.centralwidget)
        self.showresult.setGeometry(QtCore.QRect(570, 120, 501, 411))
        self.showresult.setObjectName("showresult")
        self.classification_result = QtWidgets.QLabel(self.centralwidget)
        self.classification_result.setGeometry(QtCore.QRect(630, 610, 401, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.classification_result.setFont(font)
        self.classification_result.setObjectName("classification_result")
        self.Author = QtWidgets.QLabel(self.centralwidget)
        self.Author.setGeometry(QtCore.QRect(20, 680, 151, 21))
        self.Author.setObjectName("Author")
        self.select_class = QtWidgets.QComboBox(self.centralwidget)
        self.select_class.setGeometry(QtCore.QRect(140, 520, 281, 31))
        self.select_class.setObjectName("select_class")
        self.select_item = QtWidgets.QComboBox(self.centralwidget)
        self.select_item.setGeometry(QtCore.QRect(140, 570, 281, 31))
        self.select_item.setObjectName("select_item")
        self.showdataset = QtWidgets.QGraphicsView(self.centralwidget)
        self.showdataset.setGeometry(QtCore.QRect(30, 120, 451, 381))
        self.showdataset.setObjectName("showdataset")
        self.median_filter = QtWidgets.QPushButton(self.centralwidget)
        self.median_filter.setGeometry(QtCore.QRect(570, 530, 201, 31))
        self.median_filter.setObjectName("median_filter")
        self.gaussian_filter = QtWidgets.QPushButton(self.centralwidget)
        self.gaussian_filter.setGeometry(QtCore.QRect(770, 530, 201, 31))
        self.gaussian_filter.setObjectName("gaussian_filter")
        self.LBP = QtWidgets.QPushButton(self.centralwidget)
        self.LBP.setGeometry(QtCore.QRect(570, 560, 101, 31))
        self.LBP.setObjectName("LBP")
        self.HOG = QtWidgets.QPushButton(self.centralwidget)
        self.HOG.setGeometry(QtCore.QRect(970, 530, 101, 31))
        self.HOG.setObjectName("HOG")
        self.birds_classification = QtWidgets.QPushButton(self.centralwidget)
        self.birds_classification.setGeometry(QtCore.QRect(870, 560, 201, 31))
        self.birds_classification.setObjectName("birds_classification")
        self.show_eval = QtWidgets.QPushButton(self.centralwidget)
        self.show_eval.setGeometry(QtCore.QRect(670, 560, 201, 31))
        self.show_eval.setObjectName("show_eval")
        self.show_dataset_title = QtWidgets.QLabel(self.centralwidget)
        self.show_dataset_title.setGeometry(QtCore.QRect(90, 70, 381, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(11)
        self.show_dataset_title.setFont(font)
        self.show_dataset_title.setObjectName("show_dataset_title")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(60, 520, 61, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(60, 570, 61, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1115, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.load_Image.setText(_translate("MainWindow", "Load Image"))
        self.Title.setText(_translate("MainWindow", "Pytorch CUB-birds-200 Classification"))
        self.classification_result.setText(_translate("MainWindow", "Classification result:"))
        self.Author.setText(_translate("MainWindow", "by 19200120 Guodong Chen"))
        self.median_filter.setText(_translate("MainWindow", "Image Denoising median filter"))
        self.gaussian_filter.setText(_translate("MainWindow", "Image Denoising gaussian filter"))
        self.LBP.setText(_translate("MainWindow", "LBP"))
        self.HOG.setText(_translate("MainWindow", "HOG"))
        self.birds_classification.setText(_translate("MainWindow", "Birds Classification"))
        self.show_eval.setText(_translate("MainWindow", "Show model information"))
        self.show_dataset_title.setText(_translate("MainWindow", "Browse, Query, and Retrieve the Database by Category"))
        self.label.setText(_translate("MainWindow", "class:"))
        self.label_2.setText(_translate("MainWindow", "item:"))
