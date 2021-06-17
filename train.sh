# Supervised learning case


declare -a AttnTypes=( 'col' 'row' 'colrow')
for attn in "${AttnTypes[@]}";

do
declare -a StringArray=('1995_income' 'bank_marketing' 'qsar_bio' 'online_shoppers' 'blastchar' 'htru2' 'shrutime' 'spambase' 'arcene' 'arrhythmia' 'philippine' 'mnist' 'volkert' 'creditcard' 'forest' 'kdd99')

for value in "${StringArray[@]}";
do

python train.py --dataset $value  --attentiontype $attn  --active_log --run_name nopt_${attn}
done
done
