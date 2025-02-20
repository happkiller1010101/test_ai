import pandas as pd
import os
import torch
import bitsandbytes as bnb
from datasets import load_dataset
from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

import subprocess
import time

import torch

# def print_gpu_utilization():
#     print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
#     print(f"Max Memory Allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
#     print(f"Memory Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
#     print(f"Max Memory Reserved: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")
#
# # Example usage
# print_gpu_utilization()
#
#
# def monitor_gpu():
#     while True:
#         # Run the nvidia-smi command and capture the output
#         result = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)
#         # Decode the output from bytes to string
#         output = result.stdout.decode('utf-8').strip()
#         print(f"GPU Monitoring: {output}")
#         time.sleep(10)  # Adjust the interval as needed
#
# # Start GPU monitoring in a separate thread
# import threading
# monitor_thread = threading.Thread(target=monitor_gpu)
# monitor_thread.start()


torch.cuda.set_per_process_memory_fraction(0.85)  # Reduced from 0.95 to 0.85



model_name = "NousResearch/llama-2-7b-chat-hf" # use this if you have access to the official LLaMA 2 model "meta-llama/Llama-2-7b-chat-hf", though keep in mind you'll need to pass a Hugging Face key argument
dataset_name = "train.jsonl"
new_model = "llama-2-7b-custom"
lora_r = 64
lora_alpha = 16
lora_dropout = 0.1
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False
output_dir = "./results"
num_train_epochs = 1
fp16 = True
bf16 = False
per_device_train_batch_size = 1
per_device_eval_batch_size = 1
gradient_accumulation_steps = 1
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 2e-4
weight_decay = 0.001
optim = "paged_adamw_32bit"
lr_scheduler_type = "constant"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
save_steps = 15
logging_steps = 5
max_seq_length = 512
packing = False
device_map = {"": 0}


def load_and_dataset():
    # Load the dataset from a CSV file into a pandas DataFrame
    prev_examples = pd.read_csv('test_case_dataset.csv')
    print(prev_examples)

    # Initialize lists to store prompts and responses
    User_Story = []
    Scenario = []
    Given = []
    When = []
    And = []
    Then = []

    # Iterate over the rows of the DataFrame
    for index, row in prev_examples.iterrows():
        try:
            # Assuming 'User_Story', 'Scenario', 'Given', 'When', 'And', 'Then' are column names in your CSV
            User_Story.append(row['User_Story'].strip())
            Scenario.append(row['Scenario'].strip())
            Given.append(row['Given'].strip())
            When.append(row['When'].strip())
            And.append(row['And'].strip())
            Then.append(row['Then'].strip())
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            pass

    # Create a DataFrame from the lists
    df = pd.DataFrame({
        'User_Story': User_Story,
        'Scenario': Scenario,
        'Given': Given,
        'When': When,
        'And': And,
        'Then': Then
    })

    # Remove duplicates
    df_result = df.drop_duplicates()

    print('There are ' + str(len(df_result)) + ' successfully-generated examples. Here are the first few:')

    # Display the first few rows
    df.head()

    # Split the data into train and test sets, with 90% in the train set
    train_df = df.sample(frac=0.9, random_state=42)
    test_df = df.drop(train_df.index)

    # Save the dataframes to .jsonl files
    train_df.to_json('train.jsonl', orient='records', lines=True)
    test_df.to_json('test.jsonl', orient='records', lines=True)

    # Load datasets
    train_dataset = load_dataset('json', data_files='train.jsonl', split="train")
    valid_dataset = load_dataset('json', data_files='test.jsonl', split="train")

    # Print GPU utilization at critical points
    # print_gpu_utilization()

    # Preprocess dataset
    train_dataset_mapped = train_dataset.map(lambda examples: {
        'text': [
            f"[INST]\n\n{User_Story} [/INST]\n\n"
            f"Scenario: {Scenario}\n"
            f"Given: {Given}\n"
            f"When: {When}\n"
            f"And: {And}\n"
            f"Then: {Then}\n"
            for User_Story, Scenario, Given, When, And, Then in zip(
                examples['User_Story'], examples['Scenario'], examples['Given'],
                examples['When'], examples['And'], examples['Then']
            )
        ]}, batched=True)

    valid_dataset_mapped = valid_dataset.map(lambda examples: {
        'text': [
            f"[INST]\n\n{User_Story} [/INST]\n\n"
            f"Scenario: {Scenario}\n"
            f"Given: {Given}\n"
            f"When: {When}\n"
            f"And: {And}\n"
            f"Then: {Then}\n"
            for User_Story, Scenario, Given, When, And, Then in zip(
                examples['User_Story'], examples['Scenario'], examples['Given'],
                examples['When'], examples['And'], examples['Then']
            )
        ]}, batched=True)

    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        report_to="all",
        evaluation_strategy="steps",
        eval_steps=5  # Evaluate every 20 steps
    )
    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset_mapped,
        eval_dataset=valid_dataset_mapped,  # Pass validation dataset here
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=packing,
    )
    trainer.train()
    trainer.model.save_pretrained(new_model)

    # Train the model here
    # Print GPU utilization after training
    # print_gpu_utilization()

    # Cell 4: Test the model
    # Cell 4: Test the model
    logging.set_verbosity(logging.CRITICAL)

    # Adjust the test User_Story to explicitly request the output structure
    User_Story = f"[INST]\n\nAs a user, I want to manage products in my shopping basket on an e-commerce website to make future purchases. [/INST] \n\nPlease provide the following:\n\nScenario:\nGiven:\nWhen:\nAnd:\nThen:"

    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)

    result = pipe(User_Story)

    # Print the generated text, expecting the model to fill in the structured fields
    print(result[0]['generated_text'])

    # Adjust the test User_Story to explicitly request the output structure
    User_Story = (
        "[INST]As a user, I want to manage products in my shopping basket on an e-commerce website to make future purchases.[/INST]"
        "\n\nPlease provide the following:\n\n"
        "Scenario: Validate that the user can log in successfully.\n"
        "Given: The user launches and logs into the e-commerce application with <username> and <password>\n"
        "When: The user navigates to the account page.\n"
        "And: The user accesses the account dashboard.\n"
        "Then: The user should be able to view account details.\n"
    )

    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
    result = pipe(User_Story)

    # # Print the generated text, expecting the model to fill in the structured fields
    # print(result[0]['generated_text'])

    # Test with the generated text

    print(result[0]['generated_text'])

    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    torch.cuda.empty_cache()


    model_path = "llama-2-7b-custom"  # change to your preferred path

    # Reload model in FP16 and merge it with LoRA weights
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="cpu",

    )
    model = PeftModel.from_pretrained(base_model, new_model)
    model = model.merge_and_unload()

    # Move to CPU before saving
    model = model.cpu()

    # Reload tokenizer to save it
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Save the merged model
    # Move model to CPU before saving
    model = model.to('cpu')
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)


def test_model():
    model_path = "llama-2-7b-custom"
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Adjust the test User_Story to explicitly request the output structure
    User_Story = (
        "[INST]As a user, I want to search for products on the e-commerce website to find what I need.[/INST]"
        "\n\nPlease provide the following:\n\n"
        "Scenario: Validate that the user can log in successfully.\n"
        "Given: The user launches and logs into the e-commerce application with <username> and <password>\n"
        "When: The user navigates to the account page.\n"
        "And: The user accesses the account dashboard.\n"
        "Then: The user should be able to view account details.\n"
    )

    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
    result = pipe(User_Story)

    # # Print the generated text, expecting the model to fill in the structured fields
    # print(result[0]['generated_text'])

    # Test with the generated text

    print(result[0]['generated_text'])


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # Choose which function to run
    run_main = False  # Set this to True if you want to run the main function instead

    if run_main:
        print("BitsAndBytes version:", bnb.__version__)
        print(torch.cuda.is_available())
        # monitor_thread = threading.Thread(target=monitor_gpu)
        # monitor_thread.start()
        load_and_dataset()
    else:
        test_model()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
