export CUDA_VISIBLE_DEVICES=0
cp ../../NNClassifer/bin/NNCNNLabeler .
#./NNCNNLabeler -l -train ../phone_data/debug.txt -dev ../phone_data/debug.txt -test ../phone_data/debug.txt -option ../option/option.debug 
#./NNCNNLabeler -l -train ../phone_data/phone_train.txt -dev ../phone_data/phone_dev.txt -test ../phone_data/phone_test.txt -option ../option/option.debug
nohup ./NNCNNLabeler -l -train ../phone_data/phone_train.txt -dev ../phone_data/phone_dev.txt -test ../phone_data/phone_test.txt -option ../option/option.debug > ../log/log 2>&1 &
#nohup ./NNCNNLabeler -l -train ../phone_data/test10.txt -dev ../phone_data/test10.txt -test ../phone_data/test10.txt  -option ../option/option.debug > ../log/log 2>&1 &
#./NNCNNLabeler -l -train ../phone_data/test10.txt -dev ../phone_data/test10.txt -test ../phone_data/test10.txt  -option ../option/option.debug
#valgrind ./NNCNNLabeler -l -train ../phone_data/test10.txt -dev ../phone_data/test10.txt -test ../phone_data/test10.txt  -option ../option/option.debug
#tail -f ../log/log
