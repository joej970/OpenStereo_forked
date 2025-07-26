import pandas as pd
import re
import sys

def analyze_trt_csv_profile(csv_file):
    """
    Analyze TensorRT CSV profile and print top layers and stage time breakdown.
    Args:
        csv_file (str): Path to the CSV file.
    Returns:
        top_layers (pd.DataFrame): Top layers by time.
        stage_times (pd.DataFrame): Aggregated time per stage.
    """
    df = pd.read_csv(csv_file)
    df.columns = [c.strip() for c in df.columns]

    # 1. Aggregate by layer name to combine duplicates
    layer_times = df.groupby("Layer Name", as_index=False)["Time (ms)"].sum()
    top_layers = layer_times.sort_values("Time (ms)", ascending=False).head(20)

    # 2. Define stage mapping based on regex
    def get_stage(name):
        if re.search(r'/base_model/backbone/stage0', name):
            return "Stage 0"
        if re.search(r'/base_model/backbone/stage1', name):
            return "Stage 1"
        if re.search(r'/base_model/backbone/stage2', name):
            return "Stage 2"
        if re.search(r'/base_model/backbone/stage3', name):
            return "Stage 3"
        if re.search(r'/base_model/backbone/fpn_layer', name):
            return "FPN Layers"
        if re.search(r'/base_model/refine_1', name):
            return "Refine 1"
        if re.search(r'/base_model/refine_2', name):
            return "Refine 2"
        if re.search(r'/base_model/refine_3', name):
            return "Refine 3"
        return "Other"

    df["Stage"] = df["Layer Name"].apply(get_stage)

    # 3. Aggregate time per stage
    stage_times = df.groupby("Stage", as_index=False)["Time (ms)"].sum()
    stage_times = stage_times.sort_values("Time (ms)", ascending=False)

    # Display results
    print("Top Layers:")
    print(top_layers)

    print("\nStage Time Breakdown:")
    print(stage_times)

    # Optional: Save outputs
    # top_layers.to_csv("top_layers.csv", index=False)
    # stage_times.to_csv("stage_times.csv", index=False)

    return top_layers, stage_times

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_trt_csv_profile.py <csv_file>")
        sys.exit(1)
    analyze_trt_csv_profile(sys.argv[1])