import yaml

with open(r"triton/feature_extractor/config.yaml", "r", encoding="utf-8") as f:
    print(yaml.load(f, Loader=yaml.FullLoader))
