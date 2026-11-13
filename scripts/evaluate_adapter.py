#!/usr/bin/env python3
"""Bounded adapter validation on source-disjoint qualified-source pairs."""
from __future__ import annotations
import argparse
import gc
import json
import math
import re
import time
from pathlib import Path
from _bundle import ROOT, CONFIG, bundled, setup, sha256, output_path, write_json, offline_metric_assets
setup()
from tools.trajectory_adaptation import load_pairs, load_pipeline, protocol


def validate_adapter_validation_sources(adapter_report, validation_rows):
    """Require provenance for the complete C/R ancestry, not merely pair splits."""
    config = adapter_report.get('metadata', {}).get('training_config', {})
    sources = config.get('training_sources')
    if config.get('training_source_provenance_complete') is not True or not isinstance(sources, list) or not sources:
        raise ValueError('Adapter has no complete training-source provenance for independent validation')
    for key in ('source_id', 'source_sha256', 'decoded_rgb_sha256'):
        if any(not isinstance(row, dict) or not isinstance(row.get(key), str) or not row[key] for row in sources):
            raise ValueError('Malformed adapter training-source provenance')
        training = {row[key] for row in sources}
        if training.intersection(row[key] for row in validation_rows):
            raise ValueError('Adapter training and validation sources overlap: ' + key)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs',required=True,type=Path)
    parser.add_argument('--adapter',action='append',default=[],help='NAME=project-local adapter path; baseline always included')
    parser.add_argument('--kappas',nargs='+',type=float,default=[2.])
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if not args.kappas or len(set(args.kappas))!=len(args.kappas) or any(not math.isfinite(k) or k<0 or k>3.4028234663852886e38 for k in args.kappas):
        parser.error('kappas must be distinct finite nonnegative values')
    adapters={}
    for spec in args.adapter:
        if '=' not in spec:parser.error('adapter must be NAME=PATH')
        name,path=spec.split('=',1)
        if not re.fullmatch('[A-Za-z][A-Za-z0-9_-]*',name) or name=='baseline' or name in adapters:
            parser.error('invalid/duplicate adapter name')
        path=Path(path).expanduser();path=(ROOT/path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():parser.error('adapter must be an existing project-local file')
        adapters[name]=path
    plan={'pairs':str(args.pairs),'adapters':{k:str(v) for k,v in adapters.items()},'kappas':args.kappas,'output':args.output,'execute':args.execute}
    print(json.dumps(plan,indent=2),flush=True)
    if not args.execute:return
    pair_manifest,rows=load_pairs(args.pairs.resolve(),split='validation')
    output=output_path(args.output);output.mkdir(parents=True,exist_ok=False)
    import torch
    if torch.device(args.device).type == 'cuda':
        torch.cuda.set_device(args.device)
    import numpy as np
    from PIL import Image
    from grn.utils_t2iv.sr_trajectory_adapter import load_trajectory_adapter
    base_sha=sha256(bundled(CONFIG['checkpoint']))
    identities={name:{'path':str(p.relative_to(ROOT)),'sha256':sha256(p)} for name,p in adapters.items()}
    report={'status':'generating','scope':'Source-disjoint adapter validation, not full benchmark scores. The S5 base was already trained on qualified sources.',
        'pairs_sha256':sha256(args.pairs.resolve()),'base_sha256':base_sha,'adapters':identities,
        'numeric_policy':{'torch':torch.__version__,'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'cudnn_enabled':torch.backends.cudnn.enabled},
        'validation_sources':[r['source_id'] for r in rows],'records':[]}
    write_json(output/'validation.json',report)
    try:
        pipe=load_pipeline(args.device)
        base_c={k:v.detach().cpu().clone() for k,v in pipe.model.sr_round_calibrator.state_dict().items()}
        for name,path in [('baseline',None),*adapters.items()]:
            if hasattr(pipe.model,'sr_latent_residual'):del pipe.model.sr_latent_residual
            pipe.model.sr_round_calibrator.load_state_dict(base_c,strict=True)
            if path is not None:
                if sha256(path)!=identities[name]['sha256']:raise ValueError('Adapter changed during validation')
                loaded = load_trajectory_adapter(path,pipe.model,expected_base_sha256=base_sha,trainable=False)
                validate_adapter_validation_sources(loaded, rows)
            pipe.model.eval();pipe.vae.eval()
            for kappa in args.kappas:
                method=f'{name}_k{kappa!r}'
                folder=output/method;folder.mkdir()
                for row in rows:
                    started=time.monotonic()
                    image=pipe(lr_image=row['lr'],seed=row['seed'],sr_calibrator_kappa=kappa,**protocol())
                    if image.shape!=(1,512,512,3) or image.dtype!=np.uint8:raise ValueError('Invalid prediction')
                    path_out=folder/(row['pair_id']+'.png');Image.fromarray(image[0]).save(path_out)
                    report['records'].append({'method':method,'adapter':name,'kappa':kappa,'pair_id':row['pair_id'],'seed':row['seed'],
                        'hr':row['hr'],'lr_sha256':row['lr_sha256'],'prediction':str(path_out.relative_to(ROOT)),
                        'prediction_sha256':sha256(path_out),'seconds':time.monotonic()-started,'metrics':{}})
                    write_json(output/'validation.json',report)
                    print('VALIDATED',method,row['pair_id'],flush=True)
        del pipe
        gc.collect();torch.cuda.empty_cache()
        offline_metric_assets()
        import pyiqa
        for name,kwargs in [('psnry',{'crop_border':0}),('ssim',{'crop_border':0}),('lpips',{}),('dists',{})]:
            metric=pyiqa.create_metric(name,device=args.device,**kwargs)
            with torch.inference_mode():
                for record in report['records']:
                    score=float(torch.as_tensor(metric(str(ROOT/record['prediction']),record['hr'])).mean())
                    if not math.isfinite(score):raise ValueError('Nonfinite validation metric')
                    record['metrics'][name]=score
            del metric
            gc.collect();torch.cuda.empty_cache()
            write_json(output/'validation.json',report)
        groups=sorted({r['method'] for r in report['records']})
        report['means']={method:{name:sum(r['metrics'][name] for r in report['records'] if r['method']==method)/len(rows)
            for name in ('psnry','ssim','lpips','dists')} for method in groups}
        report['status']='complete'
        write_json(output/'validation.json',report)
        print(json.dumps(report['means'],indent=2),flush=True)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_json(output/'validation.json', report)
        raise



if __name__=='__main__':main()
