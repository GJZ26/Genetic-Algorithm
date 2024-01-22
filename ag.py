from PyQt5.QtWidgets import QLabel, QProgressBar
import math
import random
from decimal import Decimal
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio


class Genetic:
    def __init__(self) -> None:
        self.current_population = []
        self.individuals = []
        self.visual_out = None
        self.visual_range = None
        self.visual_jumps = None
        self.visual_points = None
        self.visual_bit_size = None
        self.visual_delta = None
        self.progress_bar = None
        self.best_avarage = []
        self.worst_avarage = []
        self.avarage = []
        self.best = ''
        self.worst = ''
        self.preserve_edges = True

    def register_visual_output(self, logs: QLabel = None, calculated_range_label: QLabel = None,
                               jumps_label: QLabel = None, poits_label: QLabel = None,
                               bit_size_label: QLabel = None, delta_label: QLabel = None,
                               progress_bar: QProgressBar = None):
        self.visual_out, self.visual_range, self.visual_jumps = logs, calculated_range_label, jumps_label
        self.visual_points, self.visual_bit_size, self.visual_delta, self.progress_bar = poits_label, bit_size_label, delta_label, progress_bar

    def load_data(self, equation: str, is_for_maximum: bool, range_x1: float, range_x2: float,
                  max_population_size: int, initial_population_size: int, generations_number: int,
                  initial_resolution: float, mutation_probability: float, gen_mutation_probability: float,
                  cross_probability: float, preserve_edges:bool):
        self.equation = self.prepare_equation(equation)
        self.is_for_maximun = bool(is_for_maximum)
        self.range_x1, self.range_x2 = int(range_x1), int(range_x2)
        self.max_population_size, self.initial_population_size = int(
            max_population_size), int(initial_population_size)
        self.generations_number, self.initial_resolution = int(
            generations_number), Decimal(initial_resolution)
        self.mutation_probability, self.gen_mutation_probability, self.cross_probability = float(
            mutation_probability), float(gen_mutation_probability), float(cross_probability)
        self.preserve_edges = bool(preserve_edges)

    def prepare_equation(self, equation):
        return equation.replace('sin', 'math.sin').replace('tan', 'math.tan').replace('cos', 'math.cos') \
            .replace('sqrt', 'math.sqrt').replace('log', 'math.log').replace('e', 'math.e').strip()

    def pruning(self):
        classes = self.generate_classes(self.current_population['record'])
        result = self.keep_random_individuals(
            classes, self.max_population_size)
        self.current_population = {'record': result,
                                   'best': self.current_population['best'], 'worst': self.current_population['worst']}

    def generate_classes(self, data):
        classes = {}
        for item in data:
            value_at_index_3 = float(item[3])
            if value_at_index_3 not in classes:
                classes[value_at_index_3] = []
            classes[value_at_index_3].append(item)
        return classes

    def keep_random_individuals(self, classes, limit):
        result = []
        if (self.preserve_edges):
            limit = limit - 2
        for _, items in classes.items():
            if len(result) < limit:
                selected_items = random.sample(
                    items, min(len(items), limit - len(result)))
                result.extend(selected_items)
        return result

    def say_to_world(self, message: str):
        if not self.visual_out:
            print('Error: No se ha registrado un label visual para salida de mensajes.')
            return
        print(message)
        self.visual_out.setText(message)

    def render_video(self):

        video_name = 'evolution.mp4'
        images_file = os.listdir('frames')
        images = []

        self.say_to_world("Generando video...")
        
        for image_file in images_file:
            image_path = os.path.join('frames', image_file)
            images.append(imageio.imread(image_path))

        output_path = video_name
        imageio.mimsave(output_path, images, fps=10)
        self.say_to_world("¡Video guardado!")

    def start_initials_calculations(self):
        if not self.equation:
            self.say_to_world('Error: El recuadro de ecuación está vacío.')
            return False

        self.range = self.range_x2 - self.range_x1
        self.jumps = int(self.range / self.initial_resolution)
        self.points = self.jumps + 1
        self.bit_size = int(math.log2(self.points))
        self.delta_resolution = round(
            Decimal((self.range / ((2**self.bit_size)-1))), 3)
        self.display_initials_calculus()
        self.best = None
        self.worst = None
        return True

    def display_initials_calculus(self):
        self.visual_range.setText(
            f'[{self.range_x1} - {self.range_x2}] = {self.range}')
        self.visual_jumps.setText(
            f'({self.range}/{self.initial_resolution}) = {self.jumps}')
        self.visual_points.setText(f'{self.jumps} + 1 = {self.points}')
        self.visual_bit_size.setText(str(self.bit_size))
        self.visual_delta.setText(
            f'({self.range})/(2^{self.bit_size})-1 = {self.range}/{(2**self.bit_size) - 1} = {self.delta_resolution}')

    def start_algorithms(self):
        self.say_to_world('Realizando cálculos...')
        for generation in range(self.generations_number):
            self.simulate_next_gen(generation)
            self.update_progress_bar(generation + 1, self.generations_number)
        self.render_video()
        self.show_avarage()

    def show_avarage(self):
        x_value = list(range(1, self.generations_number + 1))
        plt.plot(x_value, self.best_avarage, label="Best",
                 color='#0CFF00', marker='o')
        plt.plot(x_value, self.worst_avarage,
                 label="Worst", color='#D81D1D', marker='s')
        plt.plot(x_value, self.avarage, label="Avarage",
                 color='#7EB3FF', marker='^')

        plt.title("Historial")
        plt.xlabel("Generación")
        plt.ylabel("Resultado")
        plt.legend()
        self.say_to_world("Generando gráfica histórica...")
        plt.show()

    def invoke_first_population(self):
        self.best_avarage = []
        self.worst_avarage = []
        self.avarage = []
        if (os.path.exists("frames") and os.path.isdir('frames')):
            self.delete_folder('frames')
        self.individuals = [self.random_binary_number()
                            for _ in range(self.initial_population_size)]

    def delete_folder(self, folder):
        for root, dirs, files in os.walk(folder, topdown=False):
            for file in files:
                file_path = os.path.join(root, file)
                os.remove(file_path)
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                os.rmdir(dir_path)
        os.rmdir(folder)

    def calculate_gen_data(self):
        self.current_population = []
        current_generation_data = []
        avarage = 0

        for individual_as_binary in self.individuals:
            individual_as_decimal = self.binary_to_decimal(
                individual_as_binary)
            x_valued = round(
                self.range_x1 + (individual_as_decimal * self.delta_resolution), 3)
            fx_valued = self.calculate_fx_value(x_valued)
            avarage += fx_valued

            current_generation_data.append(
                [individual_as_binary, individual_as_decimal, x_valued, fx_valued])

        best, worst = self.best_and_worst(current_generation_data)
        result = {'record': current_generation_data,
                  'best': best, 'worst': worst}
        self.current_population = result

        self.best_avarage.append(self.current_population['record'][best][3])
        self.worst_avarage.append(self.current_population['record'][worst][3])

        if (self.preserve_edges):
            self.preserve_best_and_worst(
                self.current_population['record'][best][0], self.current_population['record'][worst][0])

        self.avarage.append(avarage / len(self.individuals))

        return result

    def preserve_best_and_worst(self, new_best, new_worst):
        if self.best is None and self.worst is None:
            self.worst = new_worst
            self.best = new_best
            return
        current_best_in_decimal = self.binary_to_decimal(self.best)
        new_best_in_decimal = self.binary_to_decimal(new_best)

        current_worst_in_decimal = self.binary_to_decimal(self.worst)
        new_worst_in_decimal = self.binary_to_decimal(new_worst)

        self.best = new_best if new_best_in_decimal > current_best_in_decimal and self.is_for_maximun else self.best
        self.best = new_best if new_best_in_decimal < current_best_in_decimal and not self.is_for_maximun else self.best

        self.worst = new_worst if new_worst_in_decimal < current_worst_in_decimal and self.is_for_maximun else self.worst
        self.worst = new_worst if new_worst_in_decimal > current_worst_in_decimal and self.is_for_maximun else self.worst

    def calculate_fx_value(self, x_valued):
        try:
            return round(Decimal(eval(self.equation.replace('x', f'({x_valued})'))), 3)
        except:
            self.say_to_world(
                "Error: Hay un error en tu ecuación, verifique y vuelva a intentar.")

    def simulate_next_gen(self, generation):
        if generation == 0:
            self.invoke_first_population()
        else:
            self.individuals = [subarreglo[0]
                                for subarreglo in self.current_population['record']]
            if self.preserve_edges:
                self.individuals.append(self.best)
                self.individuals.append(self.worst)
            self.matching()
        self.calculate_gen_data()
        self.render_frame(self.current_population, generation)
        self.pruning()

    def matching(self):
        sorted_data = sorted(
            self.current_population['record'], key=lambda x: x[3], reverse=self.is_for_maximun)
        middle_index = len(sorted_data) // 2
        best_selection, worst_selection = sorted_data[:
                                                      middle_index], sorted_data[middle_index:]

        for dad in best_selection:
            for mom in worst_selection:
                if random.uniform(0.00, 1.00) <= self.cross_probability:
                    self.cross(dad, mom)

    def cross(self, dad, mom):
        points_to_cross = self.points_to_cross(
            random.randint(0, self.bit_size), 0, self.bit_size-1)
        child = self.exchange_information(dad[0], mom[0], points_to_cross)
        child = self.mutate(child)
        self.individuals.append(child)

    def points_to_cross(self, amount, low, high):
        if amount > (high - low + 1):
            raise ValueError(
                "No se pueden generar más números únicos de los disponibles en el rango.")
        return random.sample(range(low, high + 1), amount)

    def exchange_information(self, bin1, bin2, indexes):
        if len(bin1) != len(bin2):
            raise ValueError("Las cadenas deben tener la misma longitud.")
        result_chars = list(bin1)

        for index in indexes:
            if 0 <= index < len(bin1):
                result_chars[index] = bin2[index]
            else:
                raise ValueError("Índice fuera de rango.")

        return ''.join(result_chars)

    def mutate(self, child):
        if random.uniform(0.00, 1.00) > self.mutation_probability:
            return child

        mutated_child = ""
        for bit in child:
            if random.uniform(0.00, 1.00) <= self.gen_mutation_probability:
                mutated_child += "1" if bit == "0" else "0"
            else:
                mutated_child += bit

        return mutated_child

    def render_frame(self, values, frame_number):
        if not os.path.exists('frames') or not os.path.isdir('frames'):
            os.mkdir('frames')
        if values is None:
            print(
                f'No se pudo renderizar el frame correspondiente de la generación {frame_number +1 }.')
            return

        plt.clf()
        plt.title("Algoritmo genético.")
        plt.suptitle(f'Generación: {frame_number + 1 }')

        x_func = np.linspace(self.range_x1, self.range_x2, 100)
        eval_func = np.vectorize(lambda x: self.calculate_fx_value(x))
        y_func = eval_func(x_func)

        x_points, y_points = zip(*[(item[2], item[3])
                                 for item in values['record']])
        plt.scatter(x_points, y_points, color='#7EB3FF', label='Población')
        plt.plot(x_func, y_func, label=self.equation.replace(
            '**', '^').replace('math.', ''))

        best_index, worst_index = values['best'], values['worst']
        plt.scatter(values['record'][best_index][2], values['record']
                    [best_index][3], color='#0CFF00', label="Mejor individuo")
        plt.scatter(values['record'][worst_index][2], values['record']
                    [worst_index][3], color='#D81D1D', label="Peor individuo")

        plt.legend()
        plt.savefig(f'frames/{frame_number}.png')
        plt.clf()

    def best_and_worst(self, generation):
        if not generation:
            return [None, None]

        max_index = max(enumerate(generation), key=lambda x: x[1][3])[0]
        min_index = min(enumerate(generation), key=lambda x: x[1][3])[0]

        if self.is_for_maximun:
            return [max_index, min_index]
        else:
            return [min_index, max_index]

    def random_binary_number(self, to_decimal: bool = False):
        random_binary_generated = ''.join(
            [random.choice(['0', '1']) for _ in range(self.bit_size)])
        return self.binary_to_decimal(random_binary_generated) if to_decimal else random_binary_generated

    def binary_to_decimal(self, binary_as_string: str):
        return int(binary_as_string, 2)

    def update_progress_bar(self, current_step, target_step):
        print(f'{current_step} / {target_step}')
        print(f'Total population: {len(self.current_population["record"])}')
        progress_percentage = int((current_step / target_step) * 100)
        self.progress_bar.setValue(progress_percentage)
