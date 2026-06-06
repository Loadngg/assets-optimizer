import json
import os


class App:
    asset_path: str = ""
    output_dir: str = ""

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

        if not self.asset_path or not self.output_dir:
            raise KeyError(
                "Missing required keys 'asset_path' and 'output_dir' in config.json"
            )

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


def main():
    app = App()
    app.preparation()


if __name__ == "__main__":
    main()
