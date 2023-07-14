# Note: has word embeddings, gradient clipping and gaussian noise

if [ "$#" -ne 3 ]
then
    echo "illegal number of parameters"
    echo Usage:
    echo    "sh launch_experiments (negra|dptb|tiger_spmrl) modelname <gpu-id>"
    exit 1
fi

corpus=$1
gpu=$3


tr=../multilingual_disco_data/data/${corpus}/train.ctbk
dev=../multilingual_disco_data/data/${corpus}/dev.ctbk

dtok=../multilingual_disco_data/data/${corpus}/dev.tokens
ttok=../multilingual_disco_data/data/${corpus}/test.tokens

dgold=../multilingual_disco_data/data/${corpus}/dev.discbracket
tgold=../multilingual_disco_data/data/${corpus}/test.discbracket

threads=1
iterations=100
lr=0.01
dc=1e-7
ep=4
dropo=0.2
dropb=0.2
c=100
C=100
W=400
H=200
B=1

dropout="-D ${dropo} -Q ${dropb}"
dims="-c ${c} -C ${C} -W ${W} -H ${H}"
hyper="-t ${threads} -i ${iterations} -d ${dc}  -E ${ep} ${dims} -O asgd -G 100 -I 0.1"

mkdir -p ${2}

args="${tr} ${dev} ${hyper}"

(
python sfparser.py train ${2} ${args} -l 0.01 -B 1 --gpu ${gpu} -w 32 -D 0.2 -K 0 --dyno 0.15 -s 10 -T "[['tag'],['supertag'],['parsing']]" > ${2}/log.txt 2> ${2}/err.txt &&
python sfparser.py eval ${2} ${dtok} ${2}/dev_pred.discbracket  --gpu ${gpu} -t 2 --gold ${dgold} > ${2}/eval_dev &&
python sfparser.py eval ${2} ${ttok} ${2}/test_pred.discbracket --gpu ${gpu} -t 2 --gold ${tgold} > ${2}/eval_test  ) &
wait


