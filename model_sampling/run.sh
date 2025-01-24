echo "HuggingFace Token"
export HF_TOKEN=""

echo "temperature 0.7 - competition - train"
python model_sampling.py \
    --task_num 8 \
    --temperature 0.7 \
    --batch_number 8 \
    --batch_size 4 \
    --model_path "microsoft/Phi-3.5-mini-instruct" \
    --model_name "Phi_3_5_mini_instruct_vanilla" \
    --difficulty "competition" \
    --split "train"

echo "temperature 0.5 - competition - train"
python model_sampling.py \
    --task_num 8 \
    --temperature 0.5 \
    --batch_number 8 \
    --batch_size 4 \
    --model_path "microsoft/Phi-3.5-mini-instruct" \
    --model_name "Phi_3_5_mini_instruct_vanilla" \
    --difficulty "competition" \
    --split "train"

echo "temperature 0.3 - competition - train"
python model_sampling.py \
    --task_num 8 \
    --temperature 0.3 \
    --batch_number 8 \
    --batch_size 4 \
    --model_path "microsoft/Phi-3.5-mini-instruct" \
    --model_name "Phi_3_5_mini_instruct_vanilla" \
    --difficulty "competition" \
    --split "train"

echo "temperature 0.7 - introductory - test"
python model_sampling.py \
    --task_num 8 \
    --temperature 0.7 \
    --batch_number 8 \
    --batch_size 4 \
    --model_path "microsoft/Phi-3.5-mini-instruct" \
    --model_name "Phi_3_5_mini_instruct_vanilla" \
    --difficulty "introductory" \
    --split "test"

echo "temperature 0.5 - introductory - test"
python model_sampling.py \
    --task_num 8 \
    --temperature 0.5 \
    --batch_number 8 \
    --batch_size 4 \
    --model_path "microsoft/Phi-3.5-mini-instruct" \
    --model_name "Phi_3_5_mini_instruct_vanilla" \
    --difficulty "introductory" \
    --split "test"

echo "temperature 0.3 - introductory - test"
python model_sampling.py \
    --task_num 8 \
    --temperature 0.3 \
    --batch_number 8 \
    --batch_size 4 \
    --model_path "microsoft/Phi-3.5-mini-instruct" \
    --model_name "Phi_3_5_mini_instruct_vanilla" \
    --difficulty "introductory" \
    --split "test"

echo "temperature 0.7 - interview - test"
python model_sampling.py \
    --task_num 8 \
    --temperature 0.7 \
    --batch_number 8 \
    --batch_size 4 \
    --model_path "microsoft/Phi-3.5-mini-instruct" \
    --model_name "Phi_3_5_mini_instruct_vanilla" \
    --difficulty "interview" \
    --split "test"

echo "temperature 0.5 - interview - test"
python model_sampling.py \
    --task_num 8 \
    --temperature 0.5 \
    --batch_number 8 \
    --batch_size 4 \
    --model_path "microsoft/Phi-3.5-mini-instruct" \
    --model_name "Phi_3_5_mini_instruct_vanilla" \
    --difficulty "interview" \
    --split "test"

echo "temperature 0.3 - interview - test"
python model_sampling.py \
    --task_num 8 \
    --temperature 0.3 \
    --batch_number 8 \
    --batch_size 4 \
    --model_path "microsoft/Phi-3.5-mini-instruct" \
    --model_name "Phi_3_5_mini_instruct_vanilla" \
    --difficulty "interview" \
    --split "test"
