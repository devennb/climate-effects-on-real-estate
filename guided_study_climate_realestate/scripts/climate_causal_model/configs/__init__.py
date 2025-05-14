import yaml 
import os 

def read_yaml_as_dict(
        file_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datapaths.yaml')
    ):
    """
    Reads a YAML file and returns its content as a dictionary.

    Args:
        file_path (str): The path to the YAML file.

    Returns:
        dict: A dictionary representing the YAML content, or None if an error occurs.
    """
    print(file_path)
    try:
        with open(file_path, 'r') as file:
            yaml_content = yaml.safe_load(file)
            return yaml_content
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML: {e}")
        return None
    
