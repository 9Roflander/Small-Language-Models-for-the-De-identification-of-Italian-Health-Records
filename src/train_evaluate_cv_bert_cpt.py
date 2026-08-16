"""
BERT-NER 5-fold CV, starting from the domain-adapted checkpoint produced by
pretrain_cpt_bert.py instead of the vanilla dbmdz/bert-base-italian-xxl-cased.

Identical protocol to train_evaluate_cv_bert.py in every other respect (same
5-fold split, same gold+synthetic+CRF data mix, same paper_metric evaluation)
so the two results are directly comparable -- this isolates the effect of the
CPT step from everything else.
"""
import train_evaluate_cv_bert as base

base.MODEL_NAME = "./bert_it_clinical_cpt"
base.RESULTS_PATH = "./cv_bert_cpt_results.json"
base.CV_OUTPUT_BASE = "./cv_bert_cpt_outputs"

if __name__ == "__main__":
    base.run_cv()
