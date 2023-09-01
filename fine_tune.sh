echo What is the name of this traning?
read training_name

FILE=setup_done
if [ ! -f "$FILE" ]; then
python3 -m pip install --upgrade pip
pip install awscli
aws configure

aws s3 cp s3://llama2-chat-transformed/llama2-transformed/ ./llama2-transformed/ --recursive
aws s3 cp s3://forensic-training-data/ ./training_data/ --recursive

pip install -r llama-recipes/requirements.txt
touch setup_done
fi

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

python llama-recipes/llama_finetuning.py --dataset cloudf6s_dataset --model_name ./llama2-transformed --output_dir ./llama2-finetune/save/PEFT/model --bits 8
echo Uploading the result to S3 under s3://llama-finetune-output/$training_name/
aws s3 cp ./llama2-finetune/save/PEFT/model s3://llama-finetune-output/$training_name/ --recursive
