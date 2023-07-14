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
iterations=60
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


mod0="${2}0"
mod1="${2}1"
#mod2="${2}2"
#mod3="${2}3"

mkdir -p ${mod0} ${mod1}
# ${mod2} ${mod3}

git rev-parse HEAD > ${mod0}/git_rev
git rev-parse HEAD > ${mod1}/git_rev

cp ${0} ${mod0}/.
cp ${0} ${mod1}/.

args="${tr} ${dev} ${hyper}"

(
python parser.py train ${mod0} ${args} -l 0.01 -B 1 --gpu ${gpu} -w 32 -D 0 -K 0 -s 10 > ${mod0}/log.txt 2> ${mod0}/err.txt &&
python parser.py eval ${mod0} ${dtok} ${mod0}/dev_pred.discbracket  --gpu ${gpu} -t 2 --gold ${dgold} > ${mod0}/eval_dev &&
python parser.py eval ${mod0} ${ttok} ${mod0}/test_pred.discbracket --gpu ${gpu} -t 2 --gold ${tgold} > ${mod0}/eval_test  ) &

(
python parser.py train ${mod1} ${args} -l 0.01 -B 1 --gpu ${gpu} -w 32 -D 0 -K 0 --dyno 0.15 -s 10 > ${mod1}/log.txt 2> ${mod1}/err.txt &&
python parser.py eval ${mod1} ${dtok} ${mod1}/dev_pred.discbracket  --gpu ${gpu} -t 2 --gold ${dgold} > ${mod1}/eval_dev &&
python parser.py eval ${mod1} ${ttok} ${mod1}/test_pred.discbracket --gpu ${gpu} -t 2 --gold ${tgold} > ${mod1}/eval_test  ) &
wait


