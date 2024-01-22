from views.main_ui import *
import random
from ag import Genetic

equation = [
    "sin(x)**3 + x + 5",
    "cos(x**2 - 2) - 3 + x**2",
    "sin(3*x**3) + x",
    "tan(x) + 2*x",
    "4 * cos(x)",
    "2 * sin(2*x) - x**2 + 1",
    "log(x**2 + 1) + 3",
    "sqrt(x) + 2*cos(x)",
    "4*sin(x) - 2*cos(2*x) + x",
    "0.5*x**3 - 2*x**2 + 4*x - 1",
    "e**x - sin(x) + 2",
    "log(2*x) + x**2 - 3",
    "tan(2*x) - cos(x) + x",
]

class MainApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)
        self.setFixedSize(self.width(), self.height())
        self.run_btn.clicked.connect(self.calculate_variables)
        self.generate.clicked.connect(self.generate_new_equation)
        self.progress.setValue(0)
        self.visual_output.setText("")
        self.local_ag = Genetic()
        self.local_ag.register_visual_output(
            self.visual_output,
            self.range_text,
            self.jumps_visual,
            self.points_visual,
            self.bits_size,
            self.increment,
            self.progress
        )

    def calculate_variables(self):
        self.local_ag.load_data(
            self.functionVal.text(),
            self.maximum.isChecked(),
            self.x1.text(),
            self.x2.text(),
            self.max_pop.text(),
            self.initial_pop.text(),
            self.gen_num.text(),
            self.initial_resolution.text(),
            self.prob_mut_ind.text(),
            self.prob_mut_gen.text(),
            self.prob_cruza.text(),
            self.keep_edge.isChecked()
        )
        if not self.local_ag.start_initials_calculations():
            return "Error: No se pudo realizar los cálculos iniciales. Abortando."
        self.local_ag.start_algorithms()
    
    def generate_new_equation(self):
        selected_equation = equation[random.randrange(0, len(equation))]
        self.functionVal.setText(selected_equation)


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainApp()
    window.show()
    app.exec_()
