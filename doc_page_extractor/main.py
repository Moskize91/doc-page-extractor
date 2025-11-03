import os
# Set environment variables to suppress verbose output before importing other libraries
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from transformers import AutoModel, AutoTokenizer
import torch
import sys
from io import StringIO
from contextlib import contextmanager
import logging
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set logging levels
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)

# Global model cache
_model = None
_tokenizer = None
_current_device = None

def load_model(use_gpu: bool):
    """
    Load DeepSeek-OCR model with caching

    Args:
        use_gpu: Whether to use GPU acceleration

    Returns:
        Tuple of (model, tokenizer, device)
    """
    global _model, _tokenizer, _current_device

    model_name = 'deepseek-ai/DeepSeek-OCR'

    # Determine device
    if use_gpu and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # Load model if not cached or device changed
    if _model is None or _current_device != device:
        _tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Load model with appropriate settings
        model_kwargs = {
            'trust_remote_code': True,
            'use_safetensors': True
        }

        # Check if flash attention is available for GPU
        if device == 'cuda':
            try:
                import flash_attn
                model_kwargs['_attn_implementation'] = 'flash_attention_2'
            except ImportError:
                model_kwargs['_attn_implementation'] = 'eager'

        _model = AutoModel.from_pretrained(model_name, **model_kwargs)
        _model = _model.eval()

        # Move to device and set dtype
        if device == 'cuda':
            _model = _model.cuda().to(torch.bfloat16)
        else:
            _model = _model.to(device)

        _current_device = device

    return _model, _tokenizer, _current_device

def get_resolution_config(resolution: str) -> dict:
    """
    Get resolution configuration based on mode

    Args:
        resolution: Resolution mode (tiny/small/base/large/gundam)

    Returns:
        Dictionary with resolution parameters
    """
    configs = {
        'tiny': {'size': (512, 512), 'tokens': 64},
        'small': {'size': (640, 640), 'tokens': 100},
        'base': {'size': (1024, 1024), 'tokens': 256},
        'large': {'size': (1280, 1280), 'tokens': 400},
        'gundam': {'dynamic': True}  # Dynamic resolution
    }

    return configs.get(resolution, configs['base'])

def main() -> None:
    """
    Main OCR function using DeepSeek-OCR

    Args:
        params: Input parameters (image, resolution, prompt, use_gpu)
        context: OOMOL context object

    Returns:
        Outputs dictionary with extracted text
    """
    params = {
        'image': "C:\\Users\\i\\Downloads\\ocr\\source.png",
        'use_gpu': True,
        'resolution': 'base',
        'prompt': '<image>\nFree OCR.',
    }
    # Validate image path
    image_path = params['image']
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Load model
    model, tokenizer, device = load_model(params['use_gpu'])

    # Get resolution config
    resolution = params['resolution']
    res_config = get_resolution_config(resolution)

    # Prepare prompt
    prompt = params['prompt']
    full_prompt = prompt

    # Create temporary output directory
    import tempfile
    output_dir = tempfile.mkdtemp()

    # Map resolution to base_size
    resolution_to_size = {
        'tiny': 512,
        'small': 640,
        'base': 1024,
        'large': 1280,
        'gundam': 1024  # gundam uses crop_mode
    }

    base_size = resolution_to_size.get(resolution, 1024)
    image_size = 640 if resolution == 'gundam' else base_size
    crop_mode = (resolution == 'gundam')

    # Run OCR inference using model.infer()
    # Note: model.infer() returns text when save_results=True and reads from output file
    model.infer(
        tokenizer,
        prompt=full_prompt,
        image_file=image_path,
        output_path=output_dir,
        base_size=base_size,
        image_size=image_size,
        crop_mode=crop_mode,
        save_results=True,  # Enable saving to get the result
        test_compress=False
    )

    # Read the result from the saved markdown file
    result_file = os.path.join(output_dir, 'result.mmd')
    text = ""

    if os.path.exists(result_file):
        with open(result_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
    else:
        text = ""

    # Save markdown to oomol-storage
    storage_dir = 'C:\\Users\\i\\Downloads\\ocr'
    os.makedirs(storage_dir, exist_ok=True)

    # Generate unique filename based on input image
    import hashlib
    image_hash = hashlib.md5(image_path.encode()).hexdigest()[:8]
    markdown_filename = f"ocr_result_{image_hash}.md"
    markdown_path = os.path.join(storage_dir, markdown_filename)

    # Write markdown content
    with open(markdown_path, 'w', encoding='utf-8') as f:
        f.write(text)

    # Clean up temporary directory
    import shutil
    shutil.rmtree(output_dir, ignore_errors=True)

    print(markdown_path)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)