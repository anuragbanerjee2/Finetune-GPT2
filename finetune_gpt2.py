from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

def finetune(model_name,text_file,model_dir,epochs=1,batch_size=8):

    # Load GPT-2 model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    # Load training dataset
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=text_file,
        block_size=128)
    
    # Create data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,mlm=False)

    # Set training arguments
    training_args = TrainingArguments(
        output_dir=model_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=10_000,
        save_total_limit=1   
    )

    # Train the model
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,  
    )

    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)