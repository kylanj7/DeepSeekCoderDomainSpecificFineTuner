import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

class ModelTester:
    def __init__(self, model_path: str = "deepseek-coder-finetuned"):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self):
        """Load the fine-tuned model"""
        print(f"Loading model from {self.model_path}...")
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/deepseek-coder-6.7b-instruct",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load fine-tuned weights
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        print("✅ Model loaded successfully!")
    
    def generate_response(self, instruction: str, input_text: str = "", max_length: int = 512):
        """Generate response to instruction"""
        # Format prompt
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part
        response = full_response[len(prompt):].strip()
        
        return response
    
    def interactive_test(self):
        """Interactive testing mode"""
        print("\n🧪 Interactive Model Testing")
        print("Type 'quit' to exit")
        print("-" * 50)
        
        while True:
            instruction = input("\n📝 Enter your instruction: ")
            
            if instruction.lower() == 'quit':
                break
            
            input_text = input("📄 Enter input (optional, press Enter to skip): ")
            
            print("\n🤖 Generating response...")
            response = self.generate_response(instruction, input_text)
            
            print(f"\n✨ Response:\n{response}")
            print("-" * 50)
    
    def run_predefined_tests(self):
        """Run some predefined tests"""
        test_cases = [
            {
                "instruction": "Explain what a Python list comprehension is",
                "input": ""
            },
            {
                "instruction": "Write a function to reverse a string",
                "input": ""
            },
            {
                "instruction": "Explain the difference between == and is in Python",
                "input": ""
            },
            {
                "instruction": "Debug this code",
                "input": "def factorial(n):\n    if n = 1:\n        return 1\n    return n * factorial(n-1)"
            }
        ]
        
        print("\n🧪 Running Predefined Tests")
        print("=" * 50)
        
        for i, test in enumerate(test_cases, 1):
            print(f"\n📝 Test {i}: {test['instruction']}")
            if test['input']:
                print(f"📄 Input: {test['input']}")
            
            response = self.generate_response(test['instruction'], test['input'])
            print(f"✨ Response:\n{response}")
            print("-" * 50)

def main():
    # Check if model exists
    if not os.path.exists("deepseek-coder-finetuned"):
        print("❌ Fine-tuned model not found!")
        print("Please run 'python deepseek_trainer.py' first to train your model.")
        return
    
    print("🚀 Testing Fine-tuned DeepSeek Coder")
    
    # Initialize tester
    tester = ModelTester()
    
    # Load model
    tester.load_model()
    
    # Choose testing mode
    print("\nChoose testing mode:")
    print("1. Run predefined tests")
    print("2. Interactive testing")
    
    choice = input("Enter choice (1 or 2): ")
    
    if choice == "1":
        tester.run_predefined_tests()
    elif choice == "2":
        tester.interactive_test()
    else:
        print("Invalid choice. Running predefined tests...")
        tester.run_predefined_tests()
    
    print("\n✅ Testing complete!")

if __name__ == "__main__":
    main()