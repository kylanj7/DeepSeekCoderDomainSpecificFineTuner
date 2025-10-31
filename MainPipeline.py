import os
import sys
import argparse

def check_gpu():
    """Check if CUDA is available"""
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU detected: {gpu_name}")
        print(f"📊 GPU memory: {gpu_memory:.1f} GB")
        return True
    else:
        print("❌ No CUDA GPU detected!")
        print("This project requires an NVIDIA GPU with CUDA support.")
        return False

def check_directories():
    """Create necessary directories"""
    dirs = ["pdfs", "processed_data", "models"]
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"📁 Created directory: {dir_name}")

def count_pdfs():
    """Count PDFs in the pdfs directory"""
    pdf_files = [f for f in os.listdir("pdfs") if f.endswith(".pdf")]
    return len(pdf_files), pdf_files

def main():
    parser = argparse.ArgumentParser(description="DeepSeek Coder Fine-tuning Pipeline")
    parser.add_argument("--step", choices=["process", "train", "test", "all"], 
                       default="all", help="Which step to run")
    parser.add_argument("--skip-gpu-check", action="store_true", 
                       help="Skip GPU availability check")
    
    args = parser.parse_args()
    
    print("🚀 DeepSeek Coder Fine-tuning Pipeline")
    print("=" * 50)
    
    # Check GPU
    if not args.skip_gpu_check and not check_gpu():
        sys.exit(1)
    
    # Setup directories
    check_directories()
    
    # Check PDFs
    pdf_count, pdf_files = count_pdfs()
    print(f"\n📄 Found {pdf_count} PDF files in 'pdfs/' directory")
    
    if pdf_count == 0:
        print("❌ No PDF files found!")
        print("Please add your programming PDFs to the 'pdfs/' directory.")
        sys.exit(1)
    
    print("PDF files:")
    for pdf in pdf_files[:5]:  # Show first 5
        print(f"  • {pdf}")
    if pdf_count > 5:
        print(f"  ... and {pdf_count - 5} more")
    
    # Run pipeline steps
    if args.step in ["process", "all"]:
        print("\n" + "="*50)
        print("STEP 1: Processing PDFs")
        print("="*50)
        
        try:
            from PDFprocessing import PDFProcessor
            processor = PDFProcessor("pdfs/", "processed_data/")
            training_data = processor.process_all_pdfs()
            print(f"✅ Processed {len(training_data)} instruction pairs")
        except Exception as e:
            print(f"❌ PDF processing failed: {e}")
            if args.step == "all":
                sys.exit(1)
            return
    
    if args.step in ["train", "all"]:
        print("\n" + "="*50)
        print("STEP 2: Fine-tuning Model")
        print("="*50)
        
        if not os.path.exists("processed_data/training_data.json"):
            print("❌ No training data found! Run processing step first.")
            sys.exit(1)
        
        print("⚠️  Training will take several hours on RTX 3090")
        if args.step == "train":
            confirm = input("Continue? (y/n): ")
            if confirm.lower() != 'y':
                return
        
        try:
            from FineTuning import DeepSeekFineTuner
            trainer = DeepSeekFineTuner()
            trainer.load_model_and_tokenizer()
            trainer.setup_lora()
            train_data, eval_data = trainer.preprocess_data("processed_data/training_data.json")
            trainer.train(train_data, eval_data)
            print("✅ Training completed!")
        except Exception as e:
            print(f"❌ Training failed: {e}")
            if args.step == "all":
                sys.exit(1)
            return
    
    if args.step in ["test", "all"]:
        print("\n" + "="*50)
        print("STEP 3: Testing Model")
        print("="*50)
        
        if not os.path.exists("deepseek-coder-finetuned"):
            print("❌ No fine-tuned model found! Run training step first.")
            sys.exit(1)
        
        try:
            from ModelTesting import ModelTester
            tester = ModelTester()
            tester.load_model()
            
            print("\nRunning quick test...")
            response = tester.generate_response("Explain what Python is")
            print(f"Sample response: {response[:100]}...")
            
            print("\n✅ Model is working! Run 'python test_model.py' for interactive testing.")
        except Exception as e:
            print(f"❌ Model testing failed: {e}")
            return
    
    print("\n🎉 Pipeline completed successfully!")
    print("\nNext steps:")
    print("• Run 'python test_model.py' for interactive testing")
    print("• Monitor training with 'nvidia-smi' during fine-tuning")
    print("• Adjust hyperparameters in deepseek_trainer.py if needed")

if __name__ == "__main__":
    main()