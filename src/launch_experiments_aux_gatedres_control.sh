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
test=../multilingual_disco_data/data/${corpus}/test.ctbk

dtok=../multilingual_disco_data/data/${corpus}/dev.tokens
ttok=../multilingual_disco_data/data/${corpus}/test.tokens

dgold=../multilingual_disco_data/data/${corpus}/dev.discbracket
tgold=../multilingual_disco_data/data/${corpus}/test.discbracket

# oracle
dynamic=0.15

oracle="--dyno ${dynamic}"

# dropout
drop_char_emb=0.0
drop_char_out=0.0
drop_parser=0.2
drop_tagger=0.5

vardrop_i=0.0
vardrop_h=0.0

layer_norm=0

dropout="-K ${drop_char_emb} -Q ${drop_char_out} -D ${drop_parser} -X ${drop_tagger} -vi ${vardrop_i} -vh ${vardrop_h} -L ${layer_norm}"

# architectural
lstm_stack="[['tag'],[],['parsing']]"

residual_addition=0
residual_gated=0
residual_gated_output=1

initial_transform=1

activation_fun="tanh"

hid_layers_parser_ff=2
hid_layers_tagger_ff=0
bias_final_parser_ff=0
bias_final_tagger_ff=0

architectural="-T ${lstm_stack} -Ra ${residual_addition} -Rg ${residual_gated} -Rga ${residual_gated_output} -it ${initial_transform} -a ${activation_fun} -ph ${hid_layers_parser_ff} -th ${hid_layers_tagger_ff} -pb ${bias_final_parser_ff} -tb ${bias_final_tagger_ff}"

# supertagger pipeline
d_pipeline=0
drop_pipeline=0

pipeline="-sup ${d_pipeline} -Y ${drop_pipeline}"

# dimensions
d_char_emb=100
d_char_lstm=100
d_word_emb=32

d_sentence_lstm=400

d_hidden_ff=200

dimensions="-c ${d_char_emb} -C ${d_char_lstm} -w ${d_word_emb} -W ${d_sentence_lstm} -H ${d_hidden_ff}"

# initialisation
emb_init=0.1
seed=10

init="-I ${emb_init} -s ${seed}"

# training
iterations=100
learning_rate=0.01
momentum=0
decay=1e-7
eval_epochs=4
grad_clip_max=100
batch=1
optimizer="asgd"

hyper="-i ${iterations} -l ${learning_rate} -m ${momentum} -d ${decay} -E ${eval_epochs} -G ${grad_clip_max} -B ${batch} -O ${optimizer}"

# other (train and eval)
cpu_threads=1

other="-t ${cpu_threads}"

# other (eval)
eval_pipeline=0

other_eval="-pipeline ${eval_pipeline}"

dirs="${tr} ${dev} ${hyper}"
args="${oracle} ${dropout} ${architectural} ${pipeline} ${dimensions} ${init} ${hyper} ${other}"


mkdir -p ${2}
(
python sfparser.py train ${2} ${dirs} ${args} --gpu ${gpu} > ${2}/log.txt 2> ${2}/err.txt &&
python sfparser.py eval ${2} ${dtok} ${2}/dev_pred.discbracket ${other} --gold ${dgold} -ctbk ${dev} -split "dev" ${other_eval} > ${2}/eval_dev &&
python sfparser.py eval ${2} ${ttok} ${2}/test_pred.discbracket ${other} --gold ${tgold} -ctbk ${test} -split "test" ${other_eval} > ${2}/eval_test  ) &
wait

