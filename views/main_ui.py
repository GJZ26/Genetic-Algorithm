# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(758, 680)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Montserrat")
        MainWindow.setFont(font)
        MainWindow.setAutoFillBackground(False)
        MainWindow.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(30, 10, 286, 78))
        font = QtGui.QFont()
        font.setFamily("Montserrat")
        font.setPointSize(17)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(220, 230, 101, 20))
        font = QtGui.QFont()
        font.setFamily("Montserrat")
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(0, 200, 831, 20))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(30, 90, 101, 20))
        font = QtGui.QFont()
        font.setFamily("Montserrat")
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(30, 160, 201, 31))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.minimum = QtWidgets.QRadioButton(self.horizontalLayoutWidget)
        self.minimum.setObjectName("minimum")
        self.horizontalLayout.addWidget(self.minimum)
        self.maximum = QtWidgets.QRadioButton(self.horizontalLayoutWidget)
        self.maximum.setChecked(True)
        self.maximum.setObjectName("maximum")
        self.horizontalLayout.addWidget(self.maximum)
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(30, 120, 481, 41))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        font = QtGui.QFont()
        font.setFamily("Montserrat")
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.horizontalLayout_2.addWidget(self.label)
        self.functionVal = QtWidgets.QLineEdit(self.horizontalLayoutWidget_2)
        self.functionVal.setObjectName("functionVal")
        self.horizontalLayout_2.addWidget(self.functionVal)
        self.generate = QtWidgets.QPushButton(self.horizontalLayoutWidget_2)
        self.generate.setObjectName("generate")
        self.horizontalLayout_2.addWidget(self.generate)
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(30, 220, 101, 31))
        font = QtGui.QFont()
        font.setFamily("Montserrat")
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(50, 300, 41, 19))
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(50, 260, 21, 24))
        self.label_9.setObjectName("label_9")
        self.x1 = QtWidgets.QSpinBox(self.centralwidget)
        self.x1.setGeometry(QtCore.QRect(80, 260, 81, 24))
        self.x1.setMinimum(-999)
        self.x1.setMaximum(999)
        self.x1.setObjectName("x1")
        self.x2 = QtWidgets.QSpinBox(self.centralwidget)
        self.x2.setGeometry(QtCore.QRect(80, 300, 81, 24))
        self.x2.setMinimum(-999)
        self.x2.setMaximum(999)
        self.x2.setProperty("value", 10)
        self.x2.setObjectName("x2")
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setGeometry(QtCore.QRect(160, 210, 51, 181))
        self.line_2.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(220, 290, 121, 18))
        font = QtGui.QFont()
        font.setFamily("Montserrat")
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.max_pop = QtWidgets.QSpinBox(self.centralwidget)
        self.max_pop.setGeometry(QtCore.QRect(220, 315, 121, 21))
        font = QtGui.QFont()
        font.setFamily("Montserrat")
        font.setPointSize(8)
        self.max_pop.setFont(font)
        self.max_pop.setMinimum(-999)
        self.max_pop.setMaximum(999)
        self.max_pop.setProperty("value", 10)
        self.max_pop.setObjectName("max_pop")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(370, 290, 111, 18))
        font = QtGui.QFont()
        font.setFamily("Montserrat")
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.initial_pop = QtWidgets.QSpinBox(self.centralwidget)
        self.initial_pop.setGeometry(QtCore.QRect(370, 315, 111, 21))
        self.initial_pop.setMinimum(-999)
        self.initial_pop.setMaximum(999)
        self.initial_pop.setProperty("value", 4)
        self.initial_pop.setObjectName("initial_pop")
        self.run_btn = QtWidgets.QPushButton(self.centralwidget)
        self.run_btn.setGeometry(QtCore.QRect(650, 610, 93, 28))
        self.run_btn.setObjectName("run_btn")
        self.gen_num = QtWidgets.QSpinBox(self.centralwidget)
        self.gen_num.setGeometry(QtCore.QRect(520, 315, 171, 21))
        self.gen_num.setMinimum(-999)
        self.gen_num.setMaximum(999)
        self.gen_num.setProperty("value", 4)
        self.gen_num.setObjectName("gen_num")
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(520, 290, 171, 18))
        font = QtGui.QFont()
        font.setFamily("Montserrat")
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.line_3 = QtWidgets.QFrame(self.centralwidget)
        self.line_3.setGeometry(QtCore.QRect(-10, 380, 831, 20))
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.label_12 = QtWidgets.QLabel(self.centralwidget)
        self.label_12.setGeometry(QtCore.QRect(30, 620, 81, 16))
        self.label_12.setObjectName("label_12")
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        self.label_13.setGeometry(QtCore.QRect(30, 600, 111, 16))
        font = QtGui.QFont()
        font.setFamily("Montserrat")
        font.setBold(True)
        font.setWeight(75)
        self.label_13.setFont(font)
        self.label_13.setObjectName("label_13")
        self.label_14 = QtWidgets.QLabel(self.centralwidget)
        self.label_14.setGeometry(QtCore.QRect(160, 600, 111, 16))
        font = QtGui.QFont()
        font.setFamily("Montserrat")
        font.setBold(True)
        font.setWeight(75)
        self.label_14.setFont(font)
        self.label_14.setObjectName("label_14")
        self.label_15 = QtWidgets.QLabel(self.centralwidget)
        self.label_15.setGeometry(QtCore.QRect(160, 620, 211, 16))
        self.label_15.setObjectName("label_15")
        self.label_16 = QtWidgets.QLabel(self.centralwidget)
        self.label_16.setGeometry(QtCore.QRect(980, 340, 55, 16))
        self.label_16.setObjectName("label_16")
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        self.label_11.setGeometry(QtCore.QRect(40, 450, 61, 21))
        font = QtGui.QFont()
        font.setFamily("Montserrat")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.label_17 = QtWidgets.QLabel(self.centralwidget)
        self.label_17.setGeometry(QtCore.QRect(40, 410, 121, 31))
        font = QtGui.QFont()
        font.setFamily("Montserrat")
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.label_17.setFont(font)
        self.label_17.setObjectName("label_17")
        self.range_text = QtWidgets.QLabel(self.centralwidget)
        self.range_text.setGeometry(QtCore.QRect(100, 450, 211, 21))
        font = QtGui.QFont()
        font.setFamily("Montserrat")
        font.setPointSize(9)
        self.range_text.setFont(font)
        self.range_text.setObjectName("range_text")
        self.label_18 = QtWidgets.QLabel(self.centralwidget)
        self.label_18.setGeometry(QtCore.QRect(30, 490, 61, 21))
        font = QtGui.QFont()
        font.setFamily("Montserrat")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_18.setFont(font)
        self.label_18.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_18.setObjectName("label_18")
        self.increment = QtWidgets.QLabel(self.centralwidget)
        self.increment.setGeometry(QtCore.QRect(100, 490, 211, 21))
        font = QtGui.QFont()
        font.setFamily("Montserrat")
        font.setPointSize(9)
        self.increment.setFont(font)
        self.increment.setObjectName("increment")
        self.label_19 = QtWidgets.QLabel(self.centralwidget)
        self.label_19.setGeometry(QtCore.QRect(30, 470, 61, 21))
        font = QtGui.QFont()
        font.setFamily("Montserrat")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_19.setFont(font)
        self.label_19.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_19.setObjectName("label_19")
        self.bits_size = QtWidgets.QLabel(self.centralwidget)
        self.bits_size.setGeometry(QtCore.QRect(100, 470, 211, 21))
        font = QtGui.QFont()
        font.setFamily("Montserrat")
        font.setPointSize(9)
        self.bits_size.setFont(font)
        self.bits_size.setObjectName("bits_size")
        self.label_20 = QtWidgets.QLabel(self.centralwidget)
        self.label_20.setGeometry(QtCore.QRect(20, 340, 61, 24))
        self.label_20.setObjectName("label_20")
        self.bit_size = QtWidgets.QSpinBox(self.centralwidget)
        self.bit_size.setGeometry(QtCore.QRect(80, 340, 81, 24))
        self.bit_size.setMinimum(-999)
        self.bit_size.setMaximum(999)
        self.bit_size.setProperty("value", 5)
        self.bit_size.setObjectName("bit_size")
        self.progress = QtWidgets.QProgressBar(self.centralwidget)
        self.progress.setEnabled(True)
        self.progress.setGeometry(QtCore.QRect(530, 612, 118, 23))
        self.progress.setProperty("value", 24)
        self.progress.setObjectName("progress")
        self.visual_output = QtWidgets.QLabel(self.centralwidget)
        self.visual_output.setGeometry(QtCore.QRect(420, 580, 321, 20))
        self.visual_output.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.visual_output.setObjectName("visual_output")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Algoritmo Genético"))
        self.label_2.setText(_translate("MainWindow", "Algoritmo Genético"))
        self.label_5.setText(_translate("MainWindow", "Población"))
        self.label_6.setText(_translate("MainWindow", "Ecuación"))
        self.minimum.setText(_translate("MainWindow", "Minimizar"))
        self.maximum.setText(_translate("MainWindow", "Maximizar"))
        self.label.setText(_translate("MainWindow", "f(x) = "))
        self.generate.setText(_translate("MainWindow", "Generar nuevo"))
        self.label_7.setText(_translate("MainWindow", "Rango"))
        self.label_8.setText(_translate("MainWindow", "X2:"))
        self.label_9.setText(_translate("MainWindow", "X1:"))
        self.label_4.setText(_translate("MainWindow", "Población Máxima"))
        self.label_3.setText(_translate("MainWindow", "Población Inicial"))
        self.run_btn.setText(_translate("MainWindow", "Ejecutar"))
        self.label_10.setText(_translate("MainWindow", "Número de Generaciones"))
        self.label_12.setText(_translate("MainWindow", "A3C2M1P3"))
        self.label_13.setText(_translate("MainWindow", "Configuración"))
        self.label_14.setText(_translate("MainWindow", "Autor"))
        self.label_15.setText(_translate("MainWindow", "213358@ids.upchiapas.edu.mx"))
        self.label_16.setText(_translate("MainWindow", "bro?"))
        self.label_11.setText(_translate("MainWindow", "Rango:"))
        self.label_17.setText(_translate("MainWindow", "Parámetros"))
        self.range_text.setText(_translate("MainWindow", "[0,1] = |0-1| = 1"))
        self.label_18.setText(_translate("MainWindow", "∆x:"))
        self.increment.setText(_translate("MainWindow", "(rango) / (2^5 - 1) = 0"))
        self.label_19.setText(_translate("MainWindow", "Bits:"))
        self.bits_size.setText(_translate("MainWindow", "32"))
        self.label_20.setText(_translate("MainWindow", "Bits Size"))
        self.visual_output.setText(_translate("MainWindow", "This is a sample output text :)"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())