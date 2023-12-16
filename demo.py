import argparse

from src.experiment import Experiment


def run_demo():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path')
    parser.add_argument('polygon_path')
    parser.add_argument('output_path')
    args = parser.parse_args()

    config_name = 'demo.yaml'
    experiment = Experiment(config_name)
    experiment.demo(args.video_path, args.polygon_path, args.output_path)


if __name__ == '__main__':
    run_demo()
