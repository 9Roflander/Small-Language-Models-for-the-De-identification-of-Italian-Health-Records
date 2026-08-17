"""Ad-hoc ablation run: plain (non-CPT) BERT-NER 5-fold CV with the chunking
fix in train_evaluate_cv_bert.py, writing to a separate results file so it can
be compared directly against the original cv_bert_results.json (chunking is
the only variable that changes)."""
import train_evaluate_cv_bert as base

base.RESULTS_PATH = "./cv_bert_chunked_results.json"
base.CV_OUTPUT_BASE = "./cv_bert_chunked_outputs"

if __name__ == "__main__":
    base.run_cv()
