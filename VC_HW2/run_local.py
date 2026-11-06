#!/usr/bin/env python3
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
METHODS=['full_search','three_step_search','three_step_rdo']

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--student-id',required=True)
 p.add_argument('--method',choices=['all',*METHODS],default='all')
 p.add_argument('--output-root',default='.')
 args=p.parse_args()
 if not args.student_id.isdigit():p.error('student-id must be digits')
 out=Path(args.output_root).resolve()/f'HW2_{args.student_id}'
 src=HERE/'src'
 dst=out/'src'
 if dst.resolve()==src.resolve():p.error('output must not be source directory')
 dst.mkdir(parents=True,exist_ok=True)
 for item in src.iterdir():
  if item.is_file() and item.suffix in ('.py','.md'):
   shutil.copy2(item,dst/item.name)
 (out/'report').mkdir(exist_ok=True)
 ref=HERE/'images/one_gray.png';cur=HERE/'images/two_gray.png'
 for method in (METHODS if args.method=='all' else [args.method]):
  result=out/'results/submitted'/method
  result.mkdir(parents=True,exist_ok=True)
  encoded=result/'encoded.bin';decoded=result/'decoded.png'
  encode=[sys.executable,str(dst/'encoder.py'),'--reference',str(ref),'--current',str(cur),'--method',method,'--output',str(encoded)]
  decode=[sys.executable,str(dst/'decoder.py'),'--reference',str(ref),'--input',str(encoded),'--method',method,'--output',str(decoded)]
  print('ENCODE',method,flush=True);subprocess.run(encode,cwd=dst,check=True)
  print('DECODE',method,flush=True);subprocess.run(decode,cwd=dst,check=True)
  print('RESULT',result)
 print('REMINDER: Add report/HW2_'+args.student_id+'.pdf yourself; do not put TA grading output inside submission.')

if __name__=='__main__':main()
