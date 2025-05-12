
from rouge_score import rouge_scorer
import pandas as pd
from bert_score import score

# Function to calculate ROUGE scores for a single prediction and reference
def calculate_rouge(predicted, reference):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(predicted, reference)
    return scores['rouge1'].fmeasure, scores['rougeL'].fmeasure

# Function to calculate BERTScore for a single prediction and reference
def calculate_bert_score(predicted, reference):
    P, R, F1 = score([predicted], [reference], lang="en", verbose=False)
    return F1[0].item()

# Main function to process the Excel file
if __name__ == "__main__":
    excel_file_path = '/Users/swetapattnaik/Desktop/2025/GC_Practical_Language_Processing/project/eval_llama_gemini.xlsx'
    data = pd.read_excel(excel_file_path)

    if not {'QUESTION', 'RESPONSE LLAMA', 'RESPONSE GEMINI'}.issubset(data.columns):
        raise ValueError("The Excel file must contain 'QUESTION', 'RESPONSE LLAMA', and 'RESPONSE GEMINI' columns.")

    rouge1_scores = []
    rougeL_scores = []
    bert_scores = []

    for index, row in data.iterrows():
        llama_response = row['RESPONSE LLAMA']
        gemini_response = row['RESPONSE GEMINI']

        rouge1, rougeL = calculate_rouge(llama_response, gemini_response)
        bert_score = calculate_bert_score(llama_response, gemini_response)

        rouge1_scores.append(rouge1)
        rougeL_scores.append(rougeL)
        bert_scores.append(bert_score)

    data['ROUGE-1 F1'] = rouge1_scores
    data['ROUGE-L F1'] = rougeL_scores
    data['BERTScore F1'] = bert_scores

    data.to_excel(excel_file_path, index=False)
    print("ROUGE and BERT scores have been calculated and added to the Excel file.")