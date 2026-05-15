#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, math, sys
from pathlib import Path
from typing import Any
TASK_ROOT=Path(__file__).resolve().parent
TASK_ID='t5_text_to_text_minimal'
EXPECTED_COMMIT='90dcc718148715bd8e0045ca964e15dbcfba9a1d'
SUCCESS_LEVEL='L4_cpu_config_semantic_fallback'
REQUIRED_TERMS=['translate English to German', 'Hello world', 'Hallo Welt']
REQUIRED_ROUTE_KEYS=['package', 'api', 'task_registry', 'mixture_registry', 'gin_config']
def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
def sha256_file(path: Path) -> str:
    h=hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()
def validate_artifact(artifact_dir: Path) -> dict[str, Any]:
    path=artifact_dir/'expected_artifact.json'
    payload=json.loads(path.read_text(encoding='utf-8'))
    if payload.get('task_id')!=TASK_ID: raise AssertionError('wrong task_id')
    if payload.get('success_level')!=SUCCESS_LEVEL: raise AssertionError('wrong success_level')
    if payload.get('repo',{}).get('commit')!=EXPECTED_COMMIT: raise AssertionError('repo commit mismatch')
    checks=payload.get('checks')
    if not isinstance(checks,dict) or not all(v is True for v in checks.values()): raise AssertionError({'checks':checks})
    semantic=payload.get('semantic',{}); text=' '.join(str(v) for v in semantic.values())
    missing=[term for term in REQUIRED_TERMS if term not in text]
    if missing: raise AssertionError(f'missing terms: {missing}')
    route=payload.get('route',{})
    missing_route=[key for key in REQUIRED_ROUTE_KEYS if not route.get(key)]
    if missing_route: raise AssertionError(f'missing route keys: {missing_route}')
    if payload.get('sha256',{}).get('route')!=canonical_sha256(route): raise AssertionError('route sha256 mismatch')
    if payload.get('sha256',{}).get('semantic')!=canonical_sha256(semantic): raise AssertionError('semantic sha256 mismatch')
    for values in payload.get('numeric',{}).values():
        if not all(isinstance(x,(int,float)) and math.isfinite(float(x)) for x in values): raise AssertionError('bad numeric vector')
    return {'task_id':TASK_ID,'status':'pass','success_level':SUCCESS_LEVEL,'artifact_sha256':sha256_file(path),'checks':checks,'observed':{'semantic':semantic,'repo_commit':payload['repo']['commit']}}
def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument('--artifact-dir',type=Path,default=TASK_ROOT/'artifacts'); p.add_argument('--check-only',action='store_true'); p.add_argument('--json',action='store_true'); args=p.parse_args()
    try: result=validate_artifact(args.artifact_dir)
    except Exception as exc:
        print(json.dumps({'task_id':TASK_ID,'status':'fail','error':str(exc)},indent=2,sort_keys=True),file=sys.stderr); return 1
    print(json.dumps(result,indent=2,sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
