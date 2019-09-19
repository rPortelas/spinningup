#!/usr/bin/env bash
for i in {0..20}
do
   python dummy_env.py $i &
done
wait
