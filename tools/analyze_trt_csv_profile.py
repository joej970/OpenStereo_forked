import pandas as pd
import re
import sys
import json
from collections import defaultdict

def convert_trt_csv_to_perfetto_json(csv_file, output_json=None):
    """
    Convert TensorRT CSV profile to Perfetto-compatible JSON format.
    
    Args:
        csv_file (str): Path to the TensorRT CSV file
        output_json (str, optional): Path to save the JSON file
    
    Returns:
        dict: Nested dictionary representing the profiling data
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    df.columns = [c.strip() for c in df.columns]

    # Function to normalize layer names
    def normalize_layer_name(layer_name):
        # Handle Identity layers - remove the number suffix
        if layer_name.startswith('Identity_'):
            return 'Identity'
        
        # Handle onnx::Conv layers - remove the number suffix
        if layer_name.startswith('onnx::Conv_'):
            return 'onnx::Conv'
        
        # Handle PWN layers - extract the path inside PWN()
        # if layer_name.startswith('PWN(') and layer_name.endswith(')'):
        #     inner_path = layer_name[4:-1]  # Remove 'PWN(' and ')'
        #     parts = inner_path.split('/')
        #     # Keep the PWN prefix but normalize the path
        #     return f"PWN/{'/'.join(parts)}"
        
        # For regular paths, return as-is
        return layer_name
    
    # Initialize the nested structure
    nested_profile = defaultdict(lambda: {"time_ms": 0.0, "children": defaultdict(lambda: {"time_ms": 0.0, "children": {}})})
    
    # Process each row
    for _, row in df.iterrows():
        original_layer_name = row["layer_name"]
        time_ms = float(row["total_ms"])
        
        # Skip layers with 0 time
        if time_ms == 0.0:
            continue

        # Normalize the layer name
        layer_name = normalize_layer_name(original_layer_name)
            
        # Split the layer name by '/' to get hierarchy
        parts = layer_name.split('/')
        
        # Clean up parts - remove empty strings and common prefixes
        parts = [part.strip() for part in parts if part.strip()]
        
        # Navigate through the nested structure
        current_level = nested_profile
        path = []
        
        for i, part in enumerate(parts):
            path.append(part)
            
            if i == len(parts) - 1:  # Last part - this is the actual layer
                if part not in current_level:
                    current_level[part] = {"time_ms": 0.0, "children": {}}
                current_level[part]["time_ms"] += time_ms
                current_level[part]["layer_name"] = layer_name  # Store original name
            else:  # Intermediate parts - these are hierarchical groupings
                if part not in current_level:
                    current_level[part] = {"time_ms": 0.0, "children": {}}
                current_level[part]["time_ms"] += time_ms
                current_level = current_level[part]["children"]
    
    # Convert to regular dict for JSON serialization
    def convert_defaultdict(d):
        if isinstance(d, defaultdict):
            d = dict(d)
        for key, value in d.items():
            if isinstance(value, dict):
                if "children" in value and isinstance(value["children"], defaultdict):
                    value["children"] = convert_defaultdict(value["children"])
                d[key] = value
        return d
    
    nested_profile = convert_defaultdict(nested_profile)
    
    # Create Perfetto-compatible format
    perfetto_events = []
    event_id = 0
    
    def add_perfetto_events(node_dict, parent_name="", start_time=0):
        nonlocal event_id
        
        current_time = start_time
        
        for name, data in node_dict.items():
            duration = data["time_ms"] * 1000  # Convert to microseconds
            
            # Add begin event
            perfetto_events.append({
                "name": name,
                "cat": "inference",
                "ph": "B",  # Begin
                "ts": current_time,
                "pid": 1,
                "tid": 1,
                "args": {
                    "time_ms": data["time_ms"],
                    "original_name": data.get("layer_name", name)
                }
            })
            
            # Process children
            if "children" in data and data["children"]:
                add_perfetto_events(data["children"], name, current_time)
            
            # Add end event
            perfetto_events.append({
                "name": name,
                "cat": "inference", 
                "ph": "E",  # End
                "ts": current_time + duration,
                "pid": 1,
                "tid": 1
            })
            
            current_time += duration
            event_id += 1
    
    add_perfetto_events(nested_profile)
    
    # Create final Perfetto trace format
    perfetto_trace = {
        "traceEvents": perfetto_events,
        "displayTimeUnit": "ms",
        "meta": {
            "description": "TensorRT Layer Profiling",
            "source": csv_file
        }
    }
    
    # Save to file if specified
    if output_json:
        with open(output_json, 'w') as f:
            json.dump(perfetto_trace, f, indent=2)
        print(f"Perfetto JSON saved to {output_json}")
    
    return {
        "nested_profile": nested_profile,
        "perfetto_trace": perfetto_trace
    }

def print_nested_summary(nested_dict, indent=0):
    """
    Print a hierarchical summary of the nested profiling data.
    """
    for name, data in sorted(nested_dict.items(), key=lambda x: x[1]["time_ms"], reverse=True):
        prefix = "  " * indent
        time_ms = data["time_ms"]
        print(f"{prefix}{name}: {time_ms:.3f} ms")
        
        if "children" in data and data["children"]:
            print_nested_summary(data["children"], indent + 1)

def analyze_trt_csv_profile(csv_file):
    """
    Analyze TensorRT CSV profile and print top layers and stage time breakdown.
    Combines nested nodes together using re.search().
    Args:
        csv_file (str): Path to the CSV file.
    Returns:
        top_layers (pd.DataFrame): Top layers by time.
        stage_times (pd.DataFrame): Aggregated time per stage.
    """
    df = pd.read_csv(csv_file)
    df.columns = [c.strip() for c in df.columns]

    # 1. Aggregate by layer name to combine duplicates
    layer_times = df.groupby("layer_name", as_index=False)["total_ms"].sum()
    top_layers = layer_times.sort_values("total_ms", ascending=False).head(100)

    # 2. Define stage mapping based on regex. Use ^ to match the start of the string
    if re.search(r'exp_40', csv_file):
        model = "OneStereo"
    elif re.search(r'exp_300', csv_file):
        model = "LeanBackbone"
    elif re.search(r'exp_340', csv_file):
        model = "IINet"
    else:
        model = "Unknown"

    print(f"Identified model: {model}")

    if model == "OneStereo":
        def get_stage(name):
            # OneStereo (LightStereo) stages
            if re.search(r'^/base_model/backbone/impl', name):
                return "backbone_impl"
            if re.search(r'^/base_model/backbone/disassembly', name):
                return "backbone_disassembly"
            if re.search(r'^/base_model/backbone/assembly', name):
                return "backbone_assembly"
            if re.search(r'^/base_model/refine', name):
                return "refine"
            if re.search(r'^/base_model/stem', name):
                return "refine" # stem is part of refine stage
            if re.search(r'^/base_model/cost_agg', name):
                return "cost_agg"
            if re.search(r'^/base_model/', name):
                return name.replace('/base_model/', '')
            if re.search(r'^Identity', name):
                return "Identity"
            if re.search(r'^onnx::Conv', name):
                return "Conv"
            if re.search(r'^{ForeignNode', name):
                name = name.replace('{ForeignNode[', '')
                name = name[:-2]  # Remove trailing ']}'
                return get_stage(name)  # Recursively check the inner name
            if re.search(r'^PWN\(', name):
                name = name.replace('PWN(', '')
                name = name[:-1]  # Remove trailing ')'
                return get_stage(name)  # Recursively check the inner name
            
            return name

        # This function was custom selected to filter specific node names.
        def categorise(name):
            if re.search(r'^backbone_impl', name):
                return "fe_impl"
            if re.search(r'^backbone_disassembly', name):
                return "fe_disassembly"
            if re.search(r'^backbone_assembly', name):
                return "fe_assembly"
            if re.search(r'^Identity', name):
                return "fe_impl"
            if re.search(r'Concat_96', name): # probably related to cost volume construction
                return "cost_volume"
            if re.search(r'^Resize', name):
                return "out"
            if re.search(r'^ReduceSum_1', name):
                return "out"
            if re.search(r'^Mul_101', name):
                return "out"
            if re.search(r'^Softmax_1', name):
                return "refine"
            if re.search(r'^refine', name):
                return "refine"
            if re.search(r'^cost_agg', name):
                return "cost_agg"
            else:
                return "Uncategorized"
    
    else:
        def get_stage(name):
            # LeanBackbone stages
            if re.search(r'/base_model/feature_extraction', name):
                return "bb_feat_ext"
            # IINet stages
            if re.search(r'/base_model/matching_model', name):
                return "bb_match_model"
            # OneStereo (LightStereo) stages
            if re.search(r'/base_model/stem_2', name):
                return "stem_2"
            if re.search(r'/base_model/backbone/conv_stem', name):
                return "bb_conv_stem"
            if re.search(r'/base_model/backbone/stage0', name):
                return "bb_stage0"
            if re.search(r'/base_model/backbone/stage1', name):
                return "bb_stage1"
            if re.search(r'/base_model/backbone/stage2', name):
                return "bb_stage2"
            if re.search(r'/base_model/backbone/stage3', name):
                return "bb_stage3"
            if re.search(r'/base_model/backbone/fpn_layer1', name):
                return "bb_fpn_layer1"
            if re.search(r'/base_model/backbone/fpn_layer2', name):
                return "bb_fpn_layer2"
            if re.search(r'/base_model/backbone/fpn_layer3', name):
                return "bb_fpn_layer3"
            if re.search(r'/base_model/backbone/fpn_layer4', name):
                return "bb_fpn_layer4"
            if re.search(r'/base_model/backbone/out_conv', name):
                return "bb_out_conv"
            if re.search(r'/base_model/cost_agg/conv0', name):
                return "ca_conv0"
            if re.search(r'/base_model/cost_agg/conv1', name):
                return "ca_conv1"
            if re.search(r'/base_model/cost_agg/conv2', name):
                return "ca_conv2"
            if re.search(r'/base_model/cost_agg/conv3', name):
                return "ca_conv3"
            if re.search(r'/base_model/cost_agg/conv4', name):
                return "ca_conv4"
            if re.search(r'/base_model/cost_agg/conv5', name):
                return "ca_conv5"
            if re.search(r'/base_model/cost_agg/att4', name):
                return "ca_att4"
            if re.search(r'/base_model/cost_agg/att3', name):
                return "ca_att3"
            if re.search(r'/base_model/cost_agg/att2', name):
                return "ca_att2"
            if re.search(r'/base_model/cost_agg/att1', name):
                return "ca_att1"
            if re.search(r'/base_model/cost_agg/att0', name):
                return "ca_att0"
            if re.search(r'/base_model/refine_1', name):
                return "refine_1"
            if re.search(r'/base_model/refine_2', name):
                return "refine_2"
            if re.search(r'/base_model/refine_3', name):
                return "refine_3"
            if re.search(r'/base_model/Softmax_1', name):
                return "softmax_1"
            if re.search(r'/base_model/Softmax', name):
                return "softmax"
            if re.search(r'Identity', name):
                return "Identity"
            if re.search(r'Conv', name):
                return "Conv"
            return "Other"

        def categorise(name):
            return name

    df["Stage"] = df["layer_name"].apply(get_stage)
    df["StageCategorised"] = df["Stage"].apply(categorise)
    uncategorised_times = df[df["StageCategorised"] == "Uncategorized"]

    # 3. Aggregate time per stage
    uncategorised_times = uncategorised_times.groupby("Stage", as_index=False)["total_ms"].sum()
    uncategorised_times = uncategorised_times.sort_values("total_ms", ascending=False)

    stage_times = df.groupby("Stage", as_index=False)["total_ms"].sum()
    stage_times = stage_times.sort_values("total_ms", ascending=False)

    stage_times_categorised = df.groupby("StageCategorised", as_index=False)["total_ms"].sum()
    stage_times_categorised = stage_times_categorised.sort_values("total_ms", ascending=False)

    iterations = df["calls"][0] if "calls" in df.columns else 1
    stage_times["time_per_iteration_ms"] = stage_times["total_ms"] / iterations
    stage_times_categorised["time_per_iteration_ms"] = stage_times_categorised["total_ms"] / iterations

    # Display results
    # print("Top Layers:")
    # print(top_layers)

    print(f"\nStage Time Breakdown Per {iterations} iterations:")
    print(stage_times)

    # Optional: Save outputs
    # top_layers.to_csv("top_layers.csv", index=False)
    # stage_times.to_csv("stage_times.csv", index=False)

    return top_layers, stage_times, stage_times_categorised, uncategorised_times, iterations

# if __name__ == "__main__":
#     if len(sys.argv) != 2:
#         print("Usage: python analyze_trt_csv_profile.py <csv_file>")
#         sys.exit(1)
#     analyze_trt_csv_profile(sys.argv[1])

def save_nested_profile(nested_profile, output_file):
    """
    Save the nested profile to a JSON file.
    
    Args:
        nested_profile (dict): The nested profile data.
        output_file (str): Path to save the JSON file.
    """
    with open(output_file, 'w') as f:
        json.dump(nested_profile, f, indent=2)
    print(f"Nested profile saved to {output_file}")

    # Usage example
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python script.py <csv_file>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    # Convert to nested format
    result = convert_trt_csv_to_perfetto_json(csv_file, f"{csv_file[:-4]}_perfetto.json")
    
    save_nested_profile(result["nested_profile"], f"{csv_file[:-4]}_nested.json")

