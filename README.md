# DeepSeek Coder Domain-Specific Fine-Tuner

This project enables fine-tuning of the DeepSeek Coder language model on domain-specific programming materials extracted from PDF documentation. It uses the Parameter-Efficient Fine-Tuning (PEFT) method with LoRA to efficiently adapt the model with minimal computational resources.

## Features

- Extract structured content from programming PDFs
- Fine-tune DeepSeek Coder (6.7B) with LoRA
- Interactive testing of the fine-tuned model
- Memory-optimized for consumer GPUs (works on RTX 3090)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NVIDIA GPU with CUDA support (min. 12GB VRAM, 24GB recommended)
- Hugging Face Transformers and PEFT libraries
- PyMuPDF for PDF processing

## Installation

```bash
git clone https://github.com/kylanj7/DeepSeekCoderDomainSpecificFineTuner.git
cd DeepSeekCoderDomainSpecificFineTuner

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
DeepSeekCoderDomainSpecificFineTuner/
├── PDFprocessing.py      # PDF extraction and preprocessing
├── FineTuning.py         # Model fine-tuning with LoRA
├── ModelTesting.py       # Testing and evaluation utilities
├── MainPipeline.py       # End-to-end pipeline
├── pdfs/                 # Add your PDFs here
├── processed_data/       # Extracted and formatted training data
└── deepseek-coder-finetuned/  # Output directory for fine-tuned model
```

## Usage

### Step 1: Add PDFs to the `pdfs` directory

Add programming PDFs (documentation, textbooks, etc.) to the `pdfs/` directory.

### Step 2: Run the pipeline

```bash
# Run the full pipeline (process PDFs, train, and test)
python MainPipeline.py

# Or run specific steps
python MainPipeline.py --step process  # Only process PDFs
python MainPipeline.py --step train    # Only fine-tune the model
python MainPipeline.py --step test     # Only test the model
```

### Step 3: Test the fine-tuned model

```bash
# Interactive testing
python ModelTesting.py
```

## Fine-Tuning Configuration

The `FineTuning.py` file contains the configuration for LoRA fine-tuning. Key parameters:

- **Base Model**: DeepSeek Coder 6.7B Instruct
- **LoRA Rank**: 16
- **LoRA Alpha**: 32
- **Target Modules**: Attention layers (q_proj, k_proj, v_proj, o_proj)
- **Training**: 3 epochs, batch size 1, gradient accumulation steps 8

You can modify these parameters in `FineTuning.py` based on your GPU resources and specific needs.

## PDF Processing

The `PDFprocessing.py` module extracts text from PDFs and formats it into instruction-response pairs for fine-tuning. The default format:

```
### Instruction:
Explain the following programming concept from [PDF Name]:

### Response:
[Extracted content]
```

## Model Testing

After fine-tuning, you can test your model using:
- Predefined test cases
- Interactive mode for custom prompts

## Performance Considerations

- Training requires approximately 12GB VRAM with 8-bit quantization
- Fine-tuning takes several hours on an RTX 3090
- Memory usage can be adjusted by modifying batch size and gradient accumulation steps

## Acknowledgements

This project utilizes the following technologies:
- [DeepSeek Coder](https://github.com/deepseek-ai/DeepSeek-Coder) by DeepSeek AI
- [PEFT](https://github.com/huggingface/peft) library by Hugging Face
- [PyMuPDF](https://github.com/pymupdf/PyMuPDF) for PDF processing

## License

MIT License
