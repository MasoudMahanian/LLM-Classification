import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os


class My_GPT():
    def __init__(self):
        # == Settings ==
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")

        # === Local model path ===

        # for GPT2
        MODEL_PATH = "/home/masoudmahanian/my_models/gpt2"

        # ==model files===
        print("\n model files: ")
        required_files = ['config.json', 'vocab.json', 'merges.txt']
        for f in required_files:
            full_path = os.path.join(MODEL_PATH, f)
            if os.path.exists(full_path):
                print(f"✔ {f} is available.")
            else:
                print(f"✘ {f} is not available.")

        # === loading the tokenizer and model ==
        print("\nLoading model from local path...")

        try:
        # aoad tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
            
        # add PAD token (if not present)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # load model
            self.model = GPT2LMHeadModel.from_pretrained(
                MODEL_PATH,
                local_files_only=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            ).to(self.device)
            
            self.model.eval()
            
            print(f"✔ model load sucessfully")
            print(f"✔ parametters: {sum(p.numel() for p in self.model.parameters()):,}")
            
        except Exception as e:
            print(f"✘ Error: {e}")
            exit()

        # ===generate_text====
    def generate_text(self, prompt, max_length=100, temperature=0.8, top_p=0.9, num_return=1):
        """
        GPT-2 
        
        Args:
        prompt (str): start text
        max_length (int): maximum output length
        temperature (float): creativity (0.1 = accurate, 1.0 = creative)
        top_p (float): sampling of the best tokens
        num_return (int): number of different outputs
        """
        # Tokenization
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Text generation
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=num_return,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1, 
            )
        
        # تبدیل به متن
        generated_texts = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            generated_texts.append(text)
        
        return generated_texts



if __name__ == "__main__": 

    agent = My_GPT()
    print("\n" + "=" * 25)
    print("GPT-2: ")
    print("=" * 25)

    prompts = [
        "Once upon a time, in Iran, justice reigned.",
        "Unfortunately, the current Iranian government",
        "2500 years ago, the Shah of Iran",
        "Iranian king was",
    ]



    for prompt in prompts:
        print(f"\n🔹 prompt: {prompt}")
        
        # تولید متن
        results = agent.generate_text(
            prompt=prompt,
            max_length=80,
            temperature=0.8,
            top_p=0.9,
            num_return=1
        )
        
        print(f" Output: {results[0]}")
        print("_=_" * 25)