import matplotlib.pyplot as plt
import numpy as np

from reducer.reducer import Reducer


class Optimizer:
    alpha: float = 0.5
    beta: float = 0.5

    show_chart: bool = False
    print_to_console: bool = False

    reducer: Reducer = None

    _results: np.ndarray = None
    _params: np.ndarray = None

    _pareto_results: np.ndarray = None
    _pareto_params: np.ndarray = None

    def __init__(self, alpha, beta, reducer, show_chart: bool = False, print_to_console: bool = False) -> None:
        self.alpha = alpha
        self.beta = beta

        self.reducer = reducer

        self.show_chart = show_chart
        self.print_to_console = print_to_console

    def iteration_process(self) -> None:
        p1_values = np.linspace(0.1, 1.0, 40)
        p2_values = np.linspace(0.0, 1.0, 40)

        results = []
        params = []

        for p1 in p1_values:
            for p2 in p2_values:
                _, f1, f2 = self.reducer.reduce(p1, p2)
                results.append([f1, f2])
                params.append([p1, p2])

        self._results = np.array(results)
        self._params = np.array(params)

    def pareto_filter(self):
        pareto_idx = []
        for i, res in enumerate(self._results):
            is_pareto = True
            for j, other_res in enumerate(self._results):
                if i != j:
                    if (other_res[0] <= res[0] and other_res[1] <= res[1]) and \
                            (other_res[0] < res[0] or other_res[1] < res[1]):
                        is_pareto = False
                        break
            if is_pareto:
                pareto_idx.append(i)

        pareto_results = self._results[pareto_idx]
        pareto_params = self._params[pareto_idx]

        sort_order = np.argsort(pareto_results[:, 0])

        self._pareto_results = pareto_results[sort_order]
        self._pareto_params = pareto_params[sort_order]

    def normalization(self):
        f1_min, f1_max = np.min(self._pareto_results[:, 0]), np.max(self._pareto_results[:, 0])
        f2_min, f2_max = np.min(self._pareto_results[:, 1]), np.max(self._pareto_results[:, 1])

        f1_norm = (self._pareto_results[:, 0] - f1_min) / (f1_max - f1_min + 1e-6)
        f2_norm = (self._pareto_results[:, 1] - f2_min) / (f2_max - f2_min + 1e-6)

        return f1_min, f1_max, f2_min, f2_max, f1_norm, f2_norm

    def console_output(self, best_point, best_p, f1_min, f1_max, f2_min, f2_max, f1_norm, f2_norm, best_idx,
                       j_values) -> None:
        print("\n" + "=" * 60)
        print(" СИСТЕМА ОПТИМИЗАЦИИ 3D-АССЕТОВ: ОТЧЕТ О РАБОТЕ АЛГОРИТМА ")
        print("=" * 60)

        print(f"\n[1] ВХОДНЫЕ ДАННЫЕ (ОТ НЕЙРОСЕТИ):")
        print(f"    Вес критерия сжатия (Alpha):      {self.alpha:.2f}")
        print(f"    Вес критерия качества (Beta):     {self.beta:.2f}")

        print(f"\n[2] РЕЗУЛЬТАТЫ СЕТОЧНОГО ПОИСКА (GRID SEARCH):")
        print(f"    Проанализировано комбинаций:      {len(self._results)}")
        print(f"    Найдено оптимальных решений:      {len(self._pareto_results)} (Фронт Парето)")

        print(f"\n[3] ВЫБРАННОЕ ОПТИМАЛЬНОЕ РЕШЕНИЕ:")
        print(f"    > Варьируемые параметры алгоритма компрессии:")
        print(f"      - Коэффициент децимации (P1):   {best_p[0]:.3f}")
        print(f"      - Защита топологии (P2):        {best_p[1]:.3f}")
        print(f"    > Фактические критерии эффективности:")
        print(f"      - Реальный размер (C_actual):   {best_point[0]:.3f} ({(best_point[0] * 100):.1f}% от исходного)")
        print(f"      - Ошибка геометрии (E_mesh):    {best_point[1]:.5f}")

        print(f"\n[4] ВНУТРЕННЯЯ МАТЕМАТИКА (НОРМАЛИЗАЦИЯ):")
        print(f"    Диапазон C_actual на фронте:      [{f1_min:.3f} ... {f1_max:.3f}]")
        print(f"    Диапазон E_mesh на фронте:        [{f2_min:.5f} ... {f2_max:.5f}]")
        print(f"    > Нормализованные оценки выбранной точки:")
        print(f"      - Norm C_actual:                {f1_norm[best_idx]:.3f} (Ближе к 0 - лучше)")
        print(f"      - Norm E_mesh:                  {f2_norm[best_idx]:.3f} (Ближе к 0 - лучше)")
        print(f"      - Итоговое значение штрафа (J): {j_values[best_idx]:.3f}")

        print("=" * 60)

    def chart(self, best_point, best_p) -> None:
        plt.figure(figsize=(10, 6))

        plt.scatter(self._results[:, 0], self._results[:, 1], color='lightgray', alpha=0.4,
                    label='Все варианты (Облако поиска)')
        plt.plot(self._pareto_results[:, 0], self._pareto_results[:, 1], color='red', marker='o', markersize=4,
                 label='Фронт Парето')

        label_text = f'Оптимум ИИ ($\\alpha={self.alpha}, \\beta={self.beta}$)\n$P_1={best_p[0]:.2f}, P_2={best_p[1]:.2f}$'
        plt.scatter(best_point[0], best_point[1], color='blue', s=150, zorder=5, label=label_text)

        plt.title('Многокритериальная оптимизация параметров компрессии', fontsize=14, pad=15)
        plt.xlabel('Критерий $f_1$ (Реальный размер $C_{actual}$) $\\rightarrow$ Минимизация', fontsize=11)
        plt.ylabel('Критерий $f_2$ (Ошибка геометрии $E_{mesh}$) $\\rightarrow$ Минимизация', fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=10)

        plt.tight_layout()
        plt.show()

    def run(self):
        self.iteration_process()
        self.pareto_filter()

        f1_min, f1_max, f2_min, f2_max, f1_norm, f2_norm = self.normalization()

        j_values = self.alpha * f1_norm + self.beta * f2_norm

        best_idx = int(np.argmin(j_values))

        best_point = self._pareto_results[best_idx]
        best_p = self._pareto_params[best_idx]

        if self.print_to_console:
            self.console_output(best_point, best_p, f1_min, f1_max, f2_min, f2_max, f1_norm, f2_norm, best_idx,
                                j_values)

        if self.show_chart:
            self.chart(best_point, best_p)

        return best_p[0], best_p[1]
