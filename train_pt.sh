# Semisupervised learning case

for ssly in 50 200 500
do

declare -a AttnTypes=( 'col' 'row' 'colrow')
for attn in "${AttnTypes[@]}";

do
declare -a StringArray=('1995_income' 'bank_marketing' 'qsar_bio' 'online_shoppers' 'blastchar' 'htru2' 'shrutime' 'spambase' 'arcene' 'arrhythmia' 'philippine' 'mnist' 'volkert' 'creditcard' 'forest' 'kdd99')

for value in "${StringArray[@]}";
do

python train.py --dataset $value --pretrain --pretrain_epochs 100  --ssl_avail_y $ssly --attentiontype $attn  --run_name pt_${ssly}_${attn} --active_log 

done
done
done
