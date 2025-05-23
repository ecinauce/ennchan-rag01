import torch
from transformers import BitsAndBytesConfig

def str_to_bool(value: str) -> bool:
    """Convert a string representation of truth to a boolean value."""
    return str(value).lower() in ('true', 't', 'yes', 'y', '1')


def load_quantization(config):
    """
    Load quantization configuration for the model.
    
    This function checks if quantization is enabled and CUDA is available,
    then creates the appropriate BitsAndBytesConfig for model quantization.
    
    Args:
        config: Configuration object containing quantization settings
        
    Returns:
        Dictionary with quantization_config if quantization is enabled and CUDA is available,
        otherwise an empty dictionary
    """
    if config.quantization and torch.cuda.is_available():
        print("Loading quantized model...")
        # Convert string values to boolean if needed
        load_in_4bit = str_to_bool(config.quantization_config["load_in_4bit"]) if not isinstance(config.quantization_config["load_in_4bit"], bool) else config.quantization_config["load_in_4bit"]
        bnb_4bit_use_double_quant = str_to_bool(config.quantization_config["bnb_4bit_use_double_quant"]) if not isinstance(config.quantization_config["bnb_4bit_use_double_quant"], bool) else config.quantization_config["bnb_4bit_use_double_quant"]

        quantization_config = BitsAndBytesConfig(
            load_in_4bit = load_in_4bit,
            bnb_4bit_quant_type = config.quantization_config["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype = config.quantization_config["bnb_4bit_compute_dtype"],
            bnb_4bit_use_double_quant = bnb_4bit_use_double_quant,
        )
        return {"quantization_config": quantization_config}
    else:
        if config.quantization:
            print("Quantization requested but CUDA not available. My guy, either CUDA isn't installed or you're honestly too poor to run this thing. Here's the backup implementation. Good luck!")
        else:
            print("Loading full model...")
        return {}