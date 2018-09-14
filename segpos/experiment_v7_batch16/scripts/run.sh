export CUDA_VISIBLE_DEVICES=0
cp ../../NNTranJSTagger/bin/NNJSTagger_batchdecode_linear .
#nohup ./NNJSTagger -l  -train ../data/train.ctb60.pos -dev ../data/dev.ctb60.pos -test ../data/test.ctb60.pos -option ../option/option.greedy  > ../log/log 2>&1 &
nohup ./NNJSTagger_batchdecode_linear -l -train ../data_hwc/train.ctb60.pos.hwc -dev ../data_hwc/dev.ctb60.pos.hwc -test ../data_hwc/test.ctb60.pos.hwc -option ../option/option.greedy.hwc > ../log/log 2>&1 &
#./NNJSTagger -l -train ../data_hwc/train.ctb60.pos.hwc -dev ../data_hwc/dev.ctb60.pos.hwc -test ../data_hwc/test.ctb60.pos.hwc -option ../option/option.greedy.hwc
tail -f ../log/log
