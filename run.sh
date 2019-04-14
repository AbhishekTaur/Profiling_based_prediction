rm -rf test_* train_*
for i in {1..8}
    do
        python process_data.py &
    done
wait
python main_2.py
