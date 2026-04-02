# scripts/make_limited_expert_exe.py
import argparse

#from make_limited_expert import make_limited_expert_all_toks
from make_limited_expert import make_limited_expert_all_toks, make_limited_expert_all_languages

def argument_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", required=True, type=str)
    #p.add_argument("--language", required=True, type=str)
    p.add_argument(
        "--language",
        default=None,
        type=str,
        help="If set, process only this language. If omitted, process all languages under sense/.",
    )
    p.add_argument("--num-units", required=True, type=int)
    p.add_argument("--root-dir", required=True, type=str)
    p.add_argument("--task", required=True, type=str)
    p.add_argument(
        "--out-subdir",
        default=None,
        type=str,
        help="If set, write outputs to responses_tok{k}/<out-subdir>/ instead of responses_tok{k}/",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = argument_parser()
    '''
    make_limited_expert_all_toks(
        model_name=args.model_name,
        language=args.language,
        threshold=args.num_units,
        root_dir=args.root_dir,
        task=args.task,
        out_subdir=args.out_subdir,
    )
    '''
    if args.language:
        make_limited_expert_all_toks(
            model_name=args.model_name,
            language=args.language,
            threshold=args.num_units,
            root_dir=args.root_dir,
            task=args.task,
            out_subdir=args.out_subdir,
        )
    else:
        make_limited_expert_all_languages(
            model_name=args.model_name,
            threshold=args.num_units,
            root_dir=args.root_dir,
            task=args.task,
            out_subdir=args.out_subdir,
        )
