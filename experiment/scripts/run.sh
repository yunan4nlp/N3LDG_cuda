cp ../../bin/NNCNNLabeler .
./NNCNNLabeler -l -train ../phone_data/debug.txt -dev ../phone_data/debug.txt -test ../phone_data/debug.txt -option ../option/option.debug
#nohup ./NNCNNLabeler -l -train ../phone_data/phone_train.txt -dev ../phone_data/phone_dev.txt -test ../phone_data/phone_test.txt -option ../option/option.debug > log 2>&1 &
