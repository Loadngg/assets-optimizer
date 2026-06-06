import json
import os

from fbxloader import FBXLoader

from optimization.optimization import Optimizer
from predictor.predictor import Predictor
from reducer.reducer import Reducer
from viewer.viewer import Viewer


class App:
    asset_path: str = ""
    output_dir: str = ""
    show_chart: bool = False
    print_to_console: bool = False

    trimesh_model = None

    def __init__(self):
        config_path = os.path.join("config", "config.json")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Critical error: Configuration file not found at {config_path}"
            )
        except json.JSONDecodeError:
            raise ValueError(
                f"Critical error: Failed to decode JSON in {config_path}"
            )

        self.asset_path = config_data.get("asset_path")
        self.output_dir = config_data.get("output_dir")
        self.show_chart = config_data.get("show_chart")
        self.print_to_console = config_data.get("print_to_console")

    def preparation(self):
        if not os.path.isfile(self.asset_path):
            raise FileNotFoundError(
                f"Error: Asset file not found at asset_path -> {self.asset_path}"
            )

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"Created directory: {self.output_dir}")
        elif not os.path.isdir(self.output_dir):
            raise NotADirectoryError(
                f"Error: Path {self.output_dir} exists but is not a directory"
            )

    def load_model(self):
        self.trimesh_model = FBXLoader(self.asset_path).export_trimesh()

    def _get_metadata(self):
        metadata = {
            'polygons': len(self.trimesh_model.triangles),
            'vertices': len(self.trimesh_model.vertices),
        }
        return metadata

    def prediction(self):
        metadata = self._get_metadata()
        predicted_class, weight_importance, quality_importance = Predictor().analyze_model(metadata)
        if self.print_to_console:
            print('Initial polygons:', metadata['polygons'])
            print(f"Detected Topology Class: {predicted_class}")

        return weight_importance, quality_importance

    def optimize(self):
        weight_importance, quality_importance = self.prediction()
        reducer = Reducer(self.trimesh_model.triangles)

        optimizer = Optimizer(
            alpha=weight_importance,
            beta=quality_importance,
            reducer=reducer,
            show_chart=self.show_chart,
            print_to_console=self.print_to_console,
        )
        p1, p2 = optimizer.run()
        mesh, _, _ = reducer.reduce(p1, p2)
        if self.print_to_console:
            print('Reduced polygons:', len(mesh))
        return mesh

    def run(self):
        self.preparation()
        self.load_model()
        mesh = self.optimize()
        Viewer(mesh).run()
