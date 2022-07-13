Generate training and validation files:
```bash
python -W ignore main.py helper generate_train_val 0 0 S1
```
Train the a dragon model:
```bash
python -W ignore main.py dragon train "arch_id" "model_id" S2
```
Save decoded maps:
```bash
python -W ignore main.py dragon save_decoded_map "arch_id" "model_id" S2
```