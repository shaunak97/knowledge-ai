import torch
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import numpy as np

class MemoryForgeModel:
    def __init__(
        self, 
        model_name='google/flan-t5-small', 
        device=None
    ):
        """
        Flexible AI model interface for knowledge management
        
        Args:
            model_name (str): Hugging Face model identifier
            device (str, optional): Compute device
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        except Exception as e:
            print(f"Model initialization error: {e}")
            raise

    def generate_response(
        self, 
        query: str, 
        context: str = None, 
        max_length: int = 200
    ) -> str:
        """
        Generate contextual response
        
        Args:
            query (str): User's query
            context (str, optional): Additional context
            max_length (int): Maximum response length
        
        Returns:
            str: Generated response
        """
        # Combine query and context if available
        input_text = f"Context: {context}\nQuery: {query}" if context else query
        
        inputs = self.tokenizer(
            input_text, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        ).to(self.device)
        
        outputs = self.model.generate(
            **inputs, 
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def prepare_fine_tuning_data(
        self, 
        training_data: list, 
        validation_data: list = None
    ):
        """
        Prepare dataset for fine-tuning
        
        Args:
            training_data (list): List of training examples
            validation_data (list, optional): Validation dataset
        
        Returns:
            Processed datasets
        """
        def preprocess_function(examples):
            inputs = [ex['input'] for ex in examples]
            targets = [ex['target'] for ex in examples]
            
            model_inputs = self.tokenizer(
                inputs, 
                max_length=512, 
                truncation=True
            )
            
            labels = self.tokenizer(
                targets, 
                max_length=200, 
                truncation=True
            )
            
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        train_dataset = Dataset.from_list(training_data)
        train_dataset = train_dataset.map(
            preprocess_function, 
            batched=True, 
            remove_columns=train_dataset.column_names
        )
        
        val_dataset = None
        if validation_data:
            val_dataset = Dataset.from_list(validation_data)
            val_dataset = val_dataset.map(
                preprocess_function, 
                batched=True, 
                remove_columns=val_dataset.column_names
            )
        
        return train_dataset, val_dataset

    def fine_tune(
        self, 
        train_dataset, 
        validation_dataset=None, 
        output_dir='./results',
        epochs=3
    ):
        """
        Fine-tune the model on custom dataset
        
        Args:
            train_dataset: Processed training dataset
            validation_dataset: Optional validation dataset
            output_dir (str): Directory to save fine-tuned model
            epochs (int): Number of training epochs
        """
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
        )
        
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer, 
            model=self.model
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            data_collator=data_collator,
        )
        
        trainer.train()
        trainer.save_model(output_dir)

# Usage Example
if __name__ == "__main__":
    ai_model = MemoryForgeModel()
    
    # Example fine-tuning data
    training_data = [
        {
            "input": "Explain machine learning",
            "target": "Machine learning is a subset of AI focusing on learning from data"
        }
    ]
    
    train_dataset, val_dataset = ai_model.prepare_fine_tuning_data(training_data)
    
    # Uncomment to fine-tune (requires more substantial dataset)
    # ai_model.fine_tune(train_dataset, val_dataset)
    
    # Generate response example
    response = ai_model.generate_response(
        "What is machine learning?", 
        context="AI technology that learns from data"
    )
    print("Model Response:", response)