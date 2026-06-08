import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from tqdm import tqdm
from ml.training.train_model import load_data, HashtagDataset, CONFIG

def evaluate_top_k():
    print("=======================================================")
    print("CALCULATING TOP-5 ACCURACY (Recall@5)")
    print("=======================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data using existing function
    _, val_texts, _, val_encoded, mlb = load_data()
    
    # Load tokenizer and model
    print(f"\nLoading Tokenizer: {CONFIG['model_name']}...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    
    print("\nLoading Trained Model...")
    model_path = os.path.join("d:\\hashtag-generator", ".gemini", "model_output")
    
    if os.path.exists(model_path):
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=len(mlb.classes_),
            problem_type="multi_label_classification",
            ignore_mismatched_sizes=True
        )
        print("  ✓ Successfully loaded trained weights from model_output")
    else:
        print("  ! Could not find model_output. Using untrained model for evaluation.")
        model = AutoModelForSequenceClassification.from_pretrained(
            CONFIG["model_name"],
            num_labels=len(mlb.classes_),
            problem_type="multi_label_classification",
            ignore_mismatched_sizes=True
        )
        
    model.to(device)
    model.eval()

    # Create Dataset and DataLoader
    print("\nPreparing Validation Set with Knowledge Graph...")
    val_dataset = HashtagDataset(val_texts, val_encoded, tokenizer, CONFIG["max_length"], "val")
    val_loader  = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)

    print("\nRunning Inference...")
    
    total_true_tags = 0
    total_found_in_top5 = 0
    
    total_articles = 0
    articles_with_hit = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].numpy() # Shape: (batch, num_classes)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits.cpu()
            
            # Get top 5 indices for each item in batch
            # Shape: (batch, 5)
            top5_indices = torch.topk(logits, k=5, dim=1).indices.numpy()
            
            # Compare with true labels
            for i in range(len(labels)):
                true_indices = np.where(labels[i] == 1)[0]
                
                if len(true_indices) == 0:
                    continue # Skip if no hashtags
                    
                total_articles += 1
                total_true_tags += len(true_indices)
                
                # Check how many true tags are in the predicted top 5
                found_tags = sum(1 for t in true_indices if t in top5_indices[i])
                total_found_in_top5 += found_tags
                
                # Check if at least 1 tag was found
                if found_tags > 0:
                    articles_with_hit += 1

    # Calculate final metrics
    recall_at_5 = (total_found_in_top5 / total_true_tags) * 100
    hit_rate_at_5 = (articles_with_hit / total_articles) * 100
    
    print("\n=======================================================")
    print("FINAL EVALUATION RESULTS")
    print("=======================================================")
    print(f"Total Validation Articles Checked: {total_articles:,}")
    print(f"Total Correct Hashtags in Set:   {total_true_tags:,}")
    print("-------------------------------------------------------")
    print(f"Recall@5 (Top-5 Hashtag Accuracy):   {recall_at_5:.2f}%")
    print(f"Hit Rate@5 (At least 1 correct tag): {hit_rate_at_5:.2f}%")
    print("=======================================================\n")
    
if __name__ == "__main__":
    evaluate_top_k()
