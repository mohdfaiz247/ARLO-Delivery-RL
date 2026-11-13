"""Run the five-seed validation, coverage and economic sensitivity study."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from delivery_rl.studies import run_studies,DEFAULT_SEEDS,DEFAULT_SETTINGS,StudySetting,aggregate_seeds,export_figures,write_report

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds",type=int,nargs="+",default=list(DEFAULT_SEEDS))
    parser.add_argument("--evaluation-orders",type=int,default=1000)
    parser.add_argument("--quick",action="store_true",help="Small integration check; not the academic experiment")
    parser.add_argument("--refresh-report",action="store_true",help="Rebuild summaries/charts/report from a completed run without retraining")
    args=parser.parse_args()
    if args.refresh_report:
        import json
        import pandas as pd
        folder=ROOT/"results"/"studies"
        metadata=json.loads((folder/"metadata.json").read_text(encoding="utf-8"))
        if metadata["status"]!="complete": raise RuntimeError("Wait for the study to finish before rebuilding reports")
        for source,target,groups in (("validation_runs.csv","validation_summary.csv",["policy","setting"]),
                ("final_runs.csv","final_summary.csv",["policy"]),
                ("economics_runs.csv","economics_summary.csv",["policy","factor","value"])):
            aggregate_seeds(pd.read_csv(folder/source),groups).to_csv(folder/target,index=False)
        export_figures(folder); write_report(folder)
        print((folder/"REPORT.md").read_text(encoding="utf-8"))
        sys.exit(0)
    settings=(StudySetting("smoke_reference",100,.1),StudySetting("smoke_longer",200,.03)) if args.quick else DEFAULT_SETTINGS
    seeds=(1,2) if args.quick else tuple(args.seeds)
    study_root=ROOT/".study-smoke" if args.quick else ROOT
    summary=run_studies(study_root,seeds,settings,20 if args.quick else args.evaluation_orders,
                        progress=lambda message:print(message,flush=True))
    print(summary.to_string(index=False),flush=True)
