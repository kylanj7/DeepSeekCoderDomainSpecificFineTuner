import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset, load_dataset
import json
from typing import Dict, List
import wandb

class DeepSeekFineTuner:
    def __init__(self, model_name: str = "deepseek-ai/deepseek-coder-6.7b-instruct"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
    def load_model_and_tokenizer(self):
        """Load DeepSeek Coder model and tokenizer"""
        print("Loading model and tokenizer...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with optimizations for RTX 3090
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            load_in_8bit=True  # Use 8-bit quantization to save VRAM
        )


    def setup_lora(self):
        """Configure LoRA for efficient fine-tuning"""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # Simplified modules
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        
        # CRITICAL: Enable gradients for LoRA parameters
        self.model.train()
        for name, param in self.model.named_parameters():
            if "lora_" in name.lower():
                param.requires_grad = True
        
        # Print trainable parameters for debugging
        self.model.print_trainable_parameters()
        
        # Verify we have trainable parameters
        trainable_params = [name for name, param in self.model.named_parameters() if param.requires_grad]
        print(f"Number of trainable parameters: {len(trainable_params)}")
        
        if len(trainable_params) == 0:
            raise RuntimeError("No trainable parameters found! LoRA setup failed.")
    

    def format_instruction(self, instruction: str, input_text: str, response: str) -> str:
        """Format data for DeepSeek Coder instruction format"""
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{response}"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
        
        return prompt
    
    def preprocess_data(self, data_path: str, max_length: int = 1028):
        """Load and preprocess training data"""
        print("Loading and preprocessing data...")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # Format data
        formatted_data = []
        for item in raw_data:
            formatted_text = self.format_instruction(
                item['instruction'],
                item.get('input', ''),
                item['output']
            )
            formatted_data.append(formatted_text)
        
        # Tokenize
        def tokenize_function(examples):
            tokenized = self.tokenizer(
                examples['text'],
                truncation=True,
                padding=False,
                max_length=max_length,
                return_tensors=None
            )
            return tokenized
        
        # Create dataset
        dataset = Dataset.from_dict({"text": formatted_data})
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Split into train/eval
        split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
        
        print(f"Training samples: {len(split_dataset['train'])}")
        print(f"Evaluation samples: {len(split_dataset['test'])}")
        
        return split_dataset['train'], split_dataset['test']
    
    def train(self, train_dataset, eval_dataset, output_dir: str = "deepseek-coder-finetuned"):
        """Fine-tune the model"""
        print("Starting training...")
        
        # Training arguments optimized for RTX 3090
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=1,  # Reduced from 2
            per_device_eval_batch_size=1,   # Reduced from 2
            gradient_accumulation_steps=8,   # Increased to maintain effective batch size
            warmup_steps=100,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=100,
            save_steps=500,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_pin_memory=False,
            gradient_checkpointing=False,
            report_to="wandb"
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal LM, not masked LM
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # Start training
        trainer.train()
        
        # Save final model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Training complete! Model saved to {output_dir}")

# Usage example
if __name__ == "__main__":
    # Initialize trainer
    trainer = DeepSeekFineTuner()
    
    # Load model
    trainer.load_model_and_tokenizer()
    
    # Setup LoRA
    trainer.setup_lora()
    
    # Prepare data
    train_data, eval_data = trainer.preprocess_data("processed_data/training_data.json")
    
    # Train
    trainer.train(train_data, eval_data)
