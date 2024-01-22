from views.main_ui import *
from views.dialog_ui import *
import random
import math

equation = [
    "sin(x)**3 + x + 5",
    "cos(x**2 - 2) - 3 + x**2",
    "sin(3x**3) + x",
    "tan(x) + 2x",
    "4 * cos(x)",
    "2 * sin(2x) - x**2 + 1",
    "log(x**2 + 1) + 3",
    "sqrt(x) + 2*cos(x)",
    "4*sin(x) - 2*cos(2x) + x",
    "0.5*x**3 - 2*x**2 + 4*x - 1",
    "e**x - sin(x) + 2",
    "log(2*x) + x**2 - 3",
    "tan(2*x) - cos(x) + x",
]

class DialogResult(QtWidgets.QDialog, Ui_Results):
    def __init__(self, *args,  **kwargs):
        QtWidgets.QDialog.__init__(self, *args, **kwargs)
        self.setupUi(self)

class MainApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)
        self.setFixedSize(self.width(), self.height())
        self.run_btn.clicked.connect(self.calculate_variables)
        self.generate.clicked.connect(self.generate_new_equation)
        self.range_text.setText("---")
        self.bits_size.setText("---")
        self.increment.setText("---")
        self.progress.hide()

    def calculate_variables(self):
        function = (
            self.functionVal.text()
            .replace("sin", "math.sin")
            .replace("tan", "math.tan")
            .replace("cos", "math.cos")
            .replace("sqrt", "math.sqrt")
            .replace("log", "math.log")
            .replace("e", "math.e")
        )
        is_for_maximum = self.maximum.isChecked()
        range_x1 = self.x1.text()
        range_x2 = self.x2.text()
        bit_size = int(self.bit_size.text())
        max_population_size = self.max_pop.text()
        initial_pupulation = self.initial_pop.text()
        generation_number = self.gen_num.text()
        gen_range = abs(int(range_x2) - int(range_x1))
        increment = ((gen_range) / ((2**bit_size) - 1))
        if function.strip() == "":
            print("Ingrese una función válida.")
            self.visual_output.setText("Ingrese una función válida.")
            return
        self.range_text.setText(
            f"[{range_x1}, {range_x2}] = | {range_x1} - {range_x2} | = {gen_range}"
        )
        self.bits_size.setText(str(bit_size))
        self.increment.setText(
            f"({gen_range})/(2^{bit_size})-1 = {gen_range}/{(2**bit_size) - 1} = {increment}"
        )
        dialogResult = DialogResult(self)
        dialogResult.exec()

    def generate_new_equation(self):
        selected_equation = equation[random.randrange(0, len(equation))]
        self.functionVal.setText(selected_equation)


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainApp()
    window.show()
    app.exec_()
