#!/bin/sh

echo "Start ServeCore server ..."
python3 -m parrot.serve.http_server --config_path sample_configs/core/localhost_serve_core.json --log_dir log/ --log_filename core_4_vicuna_7b.log &

sleep 1
for i in {1..4} 
do  
    echo "Start Vicuna 7B server ($i/4) ..."
    python3 -m parrot.engine.http_server \
        --config_path sample_configs/engine/vicuna-7b-v1.3.json \
        --log_dir log/ \
        --log_filename engine_vicuna_7b_server_$i.log \
        --port 900$i \
        --engine_name vicuna_7b_server_$i \
        --device cuda:$[$i-1] &
    sleep 1
done

sleep 5

echo "Successfully launched Parrot runtime system."