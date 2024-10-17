docker run -it --runtime=nvidia --gpus all \
    -v /mnt/nemar/openneuro/ds004186:/mnt/nemar/openneuro/ds004186 \
    -v .:/app \
    dtyoung/eeg-ssl python main.py \
    --seed=0 --model=VGGSSL --dataset=ds004186 --sample_window=20 \
    --window=0.5 --task=RelativePositioning --tau_pos=10 --tau_neg=10 \
    --epochs=100 --batch_size=64 --lr=0.00001 --print_every=1 --num_workers=12 \
    